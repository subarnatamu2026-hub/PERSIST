import copy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Optional, Literal

import numpy as np
import pandas as pd
import safetensors.torch
import torch
import torchvision.io as io
import tyro
from einops import rearrange
from loguru import logger
from safetensors.torch import save_file
from tqdm import tqdm

from models import vae_pixel

torch.set_grad_enabled(False)

@contextmanager
def maybe_autocast(enabled: bool, device_type="cuda", dtype=None):
    if enabled:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        # Null context: yields control without wrapping
        yield

def get_pixel_images(sha256, metadata_row, output_dir: Path):
    """Load pixel image data from mp4 files, if mp4 not found attempt loading it from data.npz."""
    data_path = output_dir / metadata_row["local_path"] / "rgb.mp4"
    if data_path.exists():
        obs_rgb, _, _ = io.read_video(data_path, pts_unit='sec')  # video: (T, H, W, 3), uint8
        return obs_rgb

    logger.warning(f"{data_path} not found, trying data.npz")
    data_path = output_dir / metadata_row["local_path"] / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")
    with np.load(data_path) as data:
        if "obs_rgb" not in data:
            raise KeyError(f"obs_rgb not found in {data_path}")
        obs_rgb = torch.from_numpy(data["obs_rgb"])  # shape (T, H, W, 3)
    return obs_rgb


def load_pixel_vae_model(model_path, model_args):
    """Load pre-trained pixel VAE model from checkpoint."""
    # Create VAE model with the correct patch_size
    vae = getattr(vae_pixel, model_args.name)(**vars(model_args))
    # Load checkpoint from safetensors
    checkpoint = safetensors.torch.load_file(model_path)

    # Strip 'vae.' prefix from keys if present
    vae_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("vae."):
            vae_state_dict[key[4:]] = value
        else:
            vae_state_dict[key] = value

    vae.load_state_dict(vae_state_dict)
    vae.eval()
    vae.cuda()
    return vae


@dataclass(kw_only=True)
class Arguments:
    """Command line arguments for pixel latent encoding."""

    model: vae_pixel.ViTVaeArgs
    """Pixel VAE model configuration"""

    output_dir: str
    """Directory containing the dataset with raw data"""

    model_path: str
    """Path to pre-trained pixel VAE checkpoint"""

    batch_size: int = 4
    """Batch size for processing timesteps"""

    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"
    """Use mixed precision for encoding (fp16, bf16, or no)"""

    save_as: Literal["npz", "safetensors"] = "safetensors"
    """File format for saving latents (npz or safetensors)"""

    instances: Optional[str] = None
    """File containing specific instances to process"""

    max_workers: int = 8
    """Maximum number of worker threads for loading and saving"""

    rank: int = 0
    """Process rank for distributed processing"""

    world_size: int = 1
    """Total number of processes"""

    debug: bool = False
    """Enable debug mode"""

    skip_existing: bool = True
    """Skip instances that have already been processed (check existing files and metadata)"""

    patch_size: int = 10
    """Patch size used by the pixel VAE (for reshaping latents to spatial dimensions)"""


if __name__ == "__main__":
    args = tyro.cli(Arguments)
    logger.remove()
    log_format = f"[RANK {args.rank}/{args.world_size - 1}] - {{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{message}}"
    logger.add(
        sys.stderr,
        format=log_format,
        level="INFO",
    )

    PartialThreadPoolExecutor = partial(
        ThreadPoolExecutor, max_workers=1 if args.debug else args.max_workers
    )

    # Convert output_dir to a path object
    output_dir = Path(args.output_dir)

    # handle dtypes
    if args.mixed_precision == "bf16":
        assert args.save_as == "safetensors", "--mixed_precision bf16 only supports safetensors for file saving"
    if args.mixed_precision == "fp16":
        autocast_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    # Load pixel VAE model
    logger.info(f"Loading pixel VAE model '{args.model.name}' from {args.model_path}")
    args.model.patch_size = args.patch_size
    logger.info(f"Model configuration: \n {args.model}")
    logger.info(f"Using patch_size: {args.patch_size}")
    vae = load_pixel_vae_model(args.model_path, args.model)

    # Create pixel latents directory for saving the generated latents
    latent_dir = output_dir / "pixel_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    # Filter instances to process
    if args.instances is not None:
        with open(args.instances, "r") as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata["sha256"].isin(instances)]
    else:
        # Only process instances that have been preprocessed
        metadata = metadata[metadata["preprocessed"]]
        # Skip already processed instances (conditional)
        if args.skip_existing and "pixel_latent_generated" in metadata.columns:
            metadata = metadata[~metadata["pixel_latent_generated"]]

    # Distributed processing
    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    metadata = metadata[start:end]
    records = []

    # Get list of instances to process
    sha256s = list(metadata["sha256"].values)
    metadata_dict = {row["sha256"]: row for _, row in metadata.iterrows()}

    # Filter out already processed instances (conditional)
    if args.skip_existing:
        for sha256 in copy.copy(sha256s):
            if (latent_dir / f"{sha256}.npz").exists():
                records.append({"sha256": sha256, "pixel_latent_generated": True})
                sha256s.remove(sha256)

    logger.info(f"Processing {len(sha256s)} instances with rank {args.rank}/{args.world_size}")

    # Processing queue
    process_queue = Queue(maxsize=args.max_workers)

    # Timing
    start_time = time.time()
    load_time = 0
    encode_time = 0
    save_time = 0

    try:
        with (
            PartialThreadPoolExecutor() as loader_executor,
            PartialThreadPoolExecutor() as saver_executor,
        ):

            def loader(sha256):
                """Load pixel data for processing."""
                try:
                    metadata_row = metadata_dict[sha256]
                    obs_rgb = get_pixel_images(sha256, metadata_row, output_dir)
                    process_queue.put((sha256, obs_rgb))
                except Exception as e:
                    logger.info(f"Error loading pixel images for {sha256}: {e}")
                    # Put None to maintain queue balance
                    process_queue.put((sha256, None))

            def saver(sha256, latents_data):
                """Save encoded latents."""
                if args.save_as == "safetensors":
                    save_file(latents_data, latent_dir / f"{sha256}.safetensors")
                elif args.save_as == "npz":
                    latents_data["mean"] = latents_data["mean"].numpy()
                    np.savez_compressed(latent_dir / f"{sha256}.npz", **latents_data)
                records.append({"sha256": sha256, "pixel_latent_generated": True})

            # Start loading
            loader_executor.map(loader, sha256s)

            # Process loaded data
            for _ in tqdm(range(len(sha256s)), desc="Encoding pixel latents"):
                sha256, images_tensor = process_queue.get()
                if images_tensor is None:
                    continue
                try:
                    # obs_rgb shape: (T, H, W, 3)
                    load_start = time.time()
                    T, H, W, C = images_tensor.shape

                    # Convert to tensor and rearrange: (T, H, W, 3) -> (T, 3, H, W)
                    images_tensor = rearrange(images_tensor, "T H W C -> T C H W").float()
                    # Normalize from [0, 255] to [-1, 1] (same as MinetestPixel dataset)
                    images_tensor = (images_tensor / 255.0) * 2.0 - 1.0
                    images_tensor = images_tensor.pin_memory().cuda(non_blocking=True)
                    load_time += time.time() - load_start

                    # Process in batches to manage memory
                    latent_means = []

                    encode_start = time.time()
                    for t_start in range(0, T, args.batch_size):
                        t_end = min(t_start + args.batch_size, T)
                        batch_images = images_tensor[t_start:t_end]  # (batch, 3, H, W)

                        # Encode to latent space
                        with torch.no_grad(), maybe_autocast(
                                enabled=args.mixed_precision in ["fp16", "bf16"],
                                dtype=autocast_dtype):
                            # Use deterministic mean
                            posterior = vae.encode(batch_images)
                            z = posterior.mode()
                            latent_means.append(z.cpu())
                    encode_time += time.time() - encode_start

                    # Prepare save data
                    save_start = time.time()

                    # Calculate spatial dimensions from original image size and patch size
                    # obs_rgb shape: (T, H, W, 3), so H=360, W=640 for this dataset
                    latent_h, latent_w = H // args.patch_size, W // args.patch_size

                    # Concatenate deterministic latents ->  (T, spatial_tokens, latent_dim)
                    latent_mean = torch.cat(latent_means, axis=0)
                    # Reshape to spatial dimensions: (T, spatial_tokens, latent_dim) -> (T, H, W, latent_dim)
                    latent_mean = rearrange(
                        latent_mean, "T (H W) D -> T H W D", H=latent_h, W=latent_w
                    )
                    latents_data = {
                        "mean": latent_mean.contiguous(),
                    }

                    # Save asynchronously
                    saver_executor.submit(saver, sha256, latents_data)
                    save_time += time.time() - save_start

                except Exception as e:
                    logger.info(f"Error encoding {sha256}: {e}")
                    continue

            saver_executor.shutdown(wait=True)

    except Exception as e:
        logger.info(f"Error during processing: {e}")

    # Print timing summary
    total_time = time.time() - start_time
    logger.info("\n=== Timing Summary ===")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Load time: {load_time:.2f}s ({load_time / total_time * 100:.1f}%)")
    logger.info(f"Encode time: {encode_time:.2f}s ({encode_time / total_time * 100:.1f}%)")
    logger.info(f"Save time: {save_time:.2f}s ({save_time / total_time * 100:.1f}%)")
    logger.info(f"Other time: {total_time - load_time - encode_time - save_time:.2f}s")
    if len(sha256s) > 0:
        logger.info(f"Average per instance: {total_time / len(sha256s):.2f}s")

    # Save processing records
    if records:
        records_df = pd.DataFrame.from_records(records)
        records_path = output_dir / f"processed_pixel_latents_{args.rank}.csv"
        records_df.to_csv(records_path, index=False)
        logger.info(f"Saved {len(records)} processing records to {records_path}")
    else:
        logger.info("No instances were processed successfully")

    logger.info("Pixel latent encoding completed!")
