import copy
import json
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
import tyro
from loguru import logger
from safetensors.torch import save_file
from tqdm import tqdm

from models import vae_voxel

torch.set_grad_enabled(False)

@contextmanager
def maybe_autocast(enabled: bool, device_type="cuda", dtype=None):
    if enabled:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        # Null context: yields control without wrapping
        yield

def get_voxel_classes(sha256, output_dir):
    """Load voxel class data from processed files."""
    voxel_path = output_dir / "voxel_classes" / f"{sha256}.npz"
    if not voxel_path.exists():
        raise FileNotFoundError(f"Voxel classes file not found: {voxel_path}")
    with np.load(voxel_path) as voxel_data:
        voxel_classes = voxel_data["node_classes"]  # Load voxel classes: shape (T, X, Y, Z)
    return voxel_classes


def load_vae_model(model_path, input_res, vocab_size, latent_channels, channels, num_res_blocks):
    """Load pre-trained VAE encoder from checkpoint."""
    # Load checkpoint from safetensors
    checkpoint = safetensors.torch.load_file(model_path)
    # Create encoder model with specified architecture
    encoder = vae_voxel.ResNet3dEncoder(
        input_x=input_res,
        input_y=input_res,
        input_z=input_res,
        in_channels=vocab_size,
        latent_channels=latent_channels,
        channels=channels,
        num_res_blocks=num_res_blocks,
    )
    # Checkpoints saved from a torch.compile'd model carry an "_orig_mod." prefix; strip it.
    checkpoint = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
    encoder.load_state_dict(checkpoint)
    encoder.eval()
    encoder.cuda()
    return encoder


@dataclass
class Arguments:
    """Command line arguments for voxel latent encoding."""

    output_dir: str
    """Directory containing the dataset with voxel_classes"""

    model_path: str
    """Path to pre-trained VAE encoder checkpoint"""

    model_name: Optional[str] = None
    """Name identifier for the model (defaults to checkpoint filename)"""

    input_resolution: int = 48
    """Input resolution of VAE"""

    in_channels: int = 2138
    """Number of input channels of the VAE (must match the voxel class vocab size of the dataset)"""

    latent_channels: int = 48
    """Number of latent channels in the VAE"""

    channels: str = "32,128,512"
    """Comma-separated list of encoder channels (e.g. '16,32,64')"""

    num_res_blocks: int = 2
    """Number of residual blocks per resolution level"""

    batch_size: int = 4
    """Batch size for processing timesteps"""

    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"
    """Use mixed precision for encoding (fp16, bf16, or no)"""

    save_as: Literal["npz", "safetensors"] = "safetensors"
    """File format for saving latents (npz or safetensors)"""

    instances: Optional[str] = None
    """File containing specific instances to process"""

    max_workers: int = 8
    """Maximum number of worker threads for loading/saving"""

    rank: int = 0
    """Process rank for distributed processing"""

    world_size: int = 1
    """Total number of processes"""

    debug: bool = False
    """Enable debug mode"""

    skip_existing: bool = True
    """Skip instances that have already been processed (check existing files and metadata)"""


if __name__ == "__main__":
    args = tyro.cli(Arguments)
    logger.remove()
    log_format = f"[RANK {args.rank}/{args.world_size - 1}] - {{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{message}}"
    logger.add(sys.stderr, format=log_format, level="INFO")

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

    # Set model name
    if args.model_name is None:
        args.model_name = Path(args.model_path).stem

    # Parse channels argument
    channels = tuple(int(x.strip()) for x in args.channels.split(","))

    # Load voxel class dictionary
    classdict_path = output_dir / "mt_voxel_classdict.json"
    if not classdict_path.exists():
        raise FileNotFoundError(f"Voxel class dictionary not found: {classdict_path}")
    with open(classdict_path, "r") as f:
        voxel_classdict = json.load(f)
    vocab_size = len(voxel_classdict["node_classes"])
    if vocab_size != args.in_channels:
        raise ValueError(f"VAE input channels {args.in_channels} does not match voxel "
                         f"class vocab size {vocab_size} of the dataset.")

    # Load VAE model
    logger.info(f"Loading VAE model from {args.model_path}")
    encoder = load_vae_model(
        args.model_path, args.input_resolution, args.in_channels, args.latent_channels, channels, args.num_res_blocks
    )

    # Create voxel latents directory for saving the generated latents
    latent_dir = output_dir / "voxel_latents"
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
        # Only process instances with voxel classes available
        metadata = metadata[metadata["voxelized"]]
        # Skip already processed instances (conditional)
        if args.skip_existing and "voxel_latent_generated" in metadata.columns:
            metadata = metadata[~metadata["voxel_latent_generated"]]

    # Distributed processing
    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    metadata = metadata[start:end]
    records = []

    # Get list of instances to process
    sha256s = list(metadata["sha256"].values)

    # Filter out already processed instances for current rank based on files (conditional)
    if args.skip_existing:
        for sha256 in copy.copy(sha256s):
            if (latent_dir / f"{sha256}.npz").exists():
                records.append({"sha256": sha256, "voxel_latent_generated": True})
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
                """Load voxel data for processing."""
                try:
                    voxel_classes = get_voxel_classes(sha256, output_dir)
                    process_queue.put((sha256, voxel_classes))
                except Exception as e:
                    logger.info(f"Error loading voxel classes for {sha256}: {e}")
                    # Put None to maintain queue balance
                    process_queue.put((sha256, None))

            def saver(sha256, latents_data):
                """Save encoded latents."""
                if args.save_as == "safetensors":
                    save_file(latents_data, latent_dir / f"{sha256}.safetensors")
                elif args.save_as == "npz":
                    latents_data["mean"] = latents_data["mean"].numpy()
                    np.savez_compressed(latent_dir / f"{sha256}.npz", **latents_data)
                records.append({"sha256": sha256, "voxel_latent_generated": True})

            # Start loading
            loader_executor.map(loader, sha256s)

            # Process loaded data
            for _ in tqdm(range(len(sha256s)), desc="Encoding voxel latents"):
                sha256, voxel_classes = process_queue.get()
                if voxel_classes is None:
                    continue
                try:
                    # voxel_classes shape: (T, X, Y, Z)
                    load_start = time.time()
                    T, X, Y, Z = voxel_classes.shape

                    # Convert to tensor (keep as class indices, encoder will one-hot internally)
                    voxel_tensor = (
                        torch.from_numpy(voxel_classes).long().pin_memory().cuda(non_blocking=True)
                    )
                    load_time += time.time() - load_start

                    # Process in batches to manage memory
                    latent_means = []
                    encode_start = time.time()
                    for t_start in range(0, T, args.batch_size):
                        t_end = min(t_start + args.batch_size, T)
                        batch_voxels = voxel_tensor[t_start:t_end]  # (batch, X, Y, Z)

                        # Encode to latent space
                        with torch.no_grad(), maybe_autocast(
                                enabled=args.mixed_precision in ["fp16", "bf16"],
                                dtype=autocast_dtype):
                            # Use deterministic mean
                            z, mean, logvar = encoder(
                                batch_voxels, sample_posterior=False, return_raw=True
                            )
                            latent_means.append(mean.cpu())
                    encode_time += time.time() - encode_start

                    # Prepare save data
                    save_start = time.time()
                    # Concatenate deterministic latents
                    latent_mean = torch.cat(latent_means, axis=0)
                    latents_data = {
                        "mean": latent_mean.contiguous(),  # (T, latent_channels, ...)
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
        records_path = output_dir / f"processed_voxel_latents_{args.rank}.csv"
        records_df.to_csv(records_path, index=False)
        logger.info(f"Saved {len(records)} processing records to {records_path}")
    else:
        logger.info("No instances were processed successfully")

    logger.info("Voxel latent encoding completed!")
