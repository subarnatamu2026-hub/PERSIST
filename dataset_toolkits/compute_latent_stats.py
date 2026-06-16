from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm


def compute_latent_statistics(
    dataset_path: Path,
    num_samples: int | None = None,
    verbose: bool = False,
    process_pixel: bool = False,
    process_voxel: bool = False,
):
    """
    Compute channel-wise mean and standard deviation for pixel and voxel latents,
    using only training samples (validation_set == False) to prevent data leakage.
    - Pixel latents: shape=(T, H, W, C)
    - Voxel latents: shape=(T, C, X, Y, Z)

    Uses a parallel (batched) Welford algorithm over per-file batches.
    """
    dataset_path = Path(dataset_path)

    # Load metadata to identify training samples
    metadata_path = dataset_path / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"No metadata.csv found at {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    metadata.set_index("sha256", inplace=True)
    train_metadata = metadata[~metadata["validation_set"]]
    if num_samples is not None:
        metadata_to_use = train_metadata.iloc[:num_samples]
    else:
        metadata_to_use = train_metadata
    sha256s = list(metadata_to_use.index)
    logger.info(f"Found {len(metadata)} total samples: {len(train_metadata)} training")
    logger.info(f"Computing statistics from {len(metadata_to_use)} training samples only")

    # Get paths (latents are stored as <sha256>.safetensors)
    pixel_files = [dataset_path / "pixel_latents" / f"{sha256}.safetensors" for sha256 in sha256s]
    voxel_files = [dataset_path / "voxel_latents" / f"{sha256}.safetensors" for sha256 in sha256s]

    # Load existing statistics if doing partial computation
    existing_stats = {}
    np_stats_path = dataset_path / "latent_stats.npz"
    if int(process_pixel) + int(process_voxel) < 2 and np_stats_path.exists():
        try:
            existing_stats = dict(np.load(np_stats_path))
            logger.info(f"Loaded existing statistics from {np_stats_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing statistics: {e}")

    # Initialize Welford's algorithm variables for pixel latents
    pixel_mean = None
    pixel_M2 = None  # Sum of squared deviations
    pixel_count = 0
    prev_pixel_mean = None
    prev_pixel_std = None
    pixel_std = None

    # Process pixel latents
    if process_pixel:
        logger.info("Processing training pixel latents...")
        for file_idx, file_path in enumerate(tqdm(pixel_files)):
            try:
                data = load_file(file_path)
                latent = data["mean"].to(dtype=torch.float32)  # Shape: (T, H, W, C)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

            if pixel_mean is None:
                num_channels = latent.shape[-1]
                pixel_mean = torch.zeros(num_channels, dtype=torch.float32)
                pixel_M2 = torch.zeros(num_channels, dtype=torch.float32)

            # Parallel Welford's algorithm over this file's values
            # Flatten spatial and temporal dimensions, keep channel dimension
            values = rearrange(latent, "t h w c -> (t h w) c")  # (T*H*W, C)
            N = values.shape[0]

            batch_mean = values.mean(dim=0)
            batch_M2 = values.var(dim=0, unbiased=False) * N
            delta = batch_mean - pixel_mean
            total = pixel_count + N

            pixel_mean += delta * (N / total)
            pixel_M2 += batch_M2 + delta * delta * (pixel_count * N / total)
            pixel_count = total

            if verbose:
                current_std = torch.sqrt(pixel_M2 / max(pixel_count - 1, 1))
                if prev_pixel_mean is not None:
                    mean_delta = torch.abs(pixel_mean - prev_pixel_mean)
                    std_delta = torch.abs(current_std - prev_pixel_std)
                    print(f"\nPixel File {file_idx + 1}/{len(pixel_files)}:")
                    print(f"  Mean: {pixel_mean} (Δ: {mean_delta})")
                    print(f"  Std:  {current_std} (Δ: {std_delta})")
                else:
                    print(f"\nPixel File {file_idx + 1}/{len(pixel_files)}:")
                    print(f"  Mean: {pixel_mean}")
                    print(f"  Std:  {current_std}")
                prev_pixel_mean = pixel_mean.clone()
                prev_pixel_std = current_std.clone()

        # Compute final pixel statistics
        if pixel_count > 1:
            pixel_mean = pixel_mean.numpy()
            pixel_var = pixel_M2 / (pixel_count - 1)  # Sample variance
            pixel_std = (torch.sqrt(pixel_var)).numpy()
        else:
            pixel_mean = pixel_mean.numpy()
            pixel_std = np.zeros_like(pixel_mean)
    else:
        # Use existing pixel statistics if available
        if "pixel_mean" in existing_stats and "pixel_std" in existing_stats:
            pixel_mean = existing_stats["pixel_mean"]
            pixel_std = existing_stats["pixel_std"]
            logger.info("Using existing pixel statistics")
        else:
            logger.warning("No existing pixel statistics found, will use zeros")
            pixel_mean = np.zeros(1)  # Placeholder
            pixel_std = np.zeros(1)

    # Initialize Welford's algorithm variables for voxel latents
    voxel_mean = None
    voxel_M2 = None
    voxel_count = 0
    prev_voxel_mean = None
    prev_voxel_std = None
    voxel_std = None

    if process_voxel:
        logger.info("Processing training voxel latents...")
        for file_idx, file_path in enumerate(tqdm(voxel_files)):
            try:
                data = load_file(file_path)
                latent = data["mean"].to(dtype=torch.float32)  # Shape: (T, C, X, Y, Z)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

            if voxel_mean is None:
                num_channels = latent.shape[1]
                voxel_mean = torch.zeros(num_channels, dtype=torch.float32)
                voxel_M2 = torch.zeros(num_channels, dtype=torch.float32)

            # Parallel Welford's algorithm; reshape to (T*X*Y*Z, C)
            values = rearrange(latent, "t c x y z -> (t x y z) c")  # (T*X*Y*Z, C)
            N = values.shape[0]

            batch_mean = values.mean(dim=0)
            batch_M2 = values.var(dim=0, unbiased=False) * N
            delta = batch_mean - voxel_mean
            total = voxel_count + N

            voxel_mean += delta * (N / total)
            voxel_M2 += batch_M2 + delta * delta * (voxel_count * N / total)
            voxel_count = total

            if verbose:
                current_std = torch.sqrt(voxel_M2 / max(voxel_count - 1, 1))
                if prev_voxel_mean is not None:
                    mean_delta = torch.abs(voxel_mean - prev_voxel_mean)
                    std_delta = torch.abs(current_std - prev_voxel_std)
                    print(f"\nVoxel File {file_idx + 1}/{len(voxel_files)}:")
                    print(f"  Mean: {voxel_mean} (Δ: {mean_delta})")
                    print(f"  Std:  {current_std} (Δ: {std_delta})")
                else:
                    print(f"\nVoxel File {file_idx + 1}/{len(voxel_files)}:")
                    print(f"  Mean: {voxel_mean}")
                    print(f"  Std:  {current_std}")
                prev_voxel_mean = voxel_mean.clone()
                prev_voxel_std = current_std.clone()

        # Compute final voxel statistics
        if voxel_count > 1:
            voxel_mean = voxel_mean.numpy()
            voxel_var = voxel_M2 / (voxel_count - 1)  # Sample variance
            voxel_std = (torch.sqrt(voxel_var)).numpy()
        else:
            voxel_mean = voxel_mean.numpy()
            voxel_std = np.zeros_like(voxel_mean)
    else:
        # Use existing voxel statistics if available
        if "voxel_mean" in existing_stats and "voxel_std" in existing_stats:
            voxel_mean = existing_stats["voxel_mean"]
            voxel_std = existing_stats["voxel_std"]
            logger.info("Using existing voxel statistics")
        else:
            logger.warning("No existing voxel statistics found, will use zeros")
            voxel_mean = np.zeros(1)  # Placeholder
            voxel_std = np.zeros(1)

    # Print results
    if process_pixel:
        print("\n" + "=" * 50)
        print("PIXEL LATENT STATISTICS")
        print("=" * 50)
        print(f"Channels: {len(pixel_mean)}")
        print(f"Total samples processed: {pixel_count}")
        print(f"Mean per channel: {pixel_mean}")
        print(f"Std per channel: {pixel_std}")

    if process_voxel:
        print("\n" + "=" * 50)
        print("VOXEL LATENT STATISTICS")
        print("=" * 50)
        print(f"Channels: {len(voxel_mean)}")
        print(f"Total samples processed: {voxel_count}")
        print(f"Mean per channel: {voxel_mean}")
        print(f"Std per channel: {voxel_std}")

    # Save statistics (preserving existing data when doing partial computation)
    save_data = {}
    if process_pixel:
        save_data["pixel_mean"] = pixel_mean
        save_data["pixel_std"] = pixel_std
    elif "pixel_mean" in existing_stats:
        save_data["pixel_mean"] = existing_stats["pixel_mean"]
        save_data["pixel_std"] = existing_stats["pixel_std"]

    if process_voxel:
        save_data["voxel_mean"] = voxel_mean
        save_data["voxel_std"] = voxel_std
    elif "voxel_mean" in existing_stats:
        save_data["voxel_mean"] = existing_stats["voxel_mean"]
        save_data["voxel_std"] = existing_stats["voxel_std"]

    np.savez(np_stats_path, **save_data)
    logger.info(f"Statistics saved to: {np_stats_path}")


@dataclass
class Arguments:
    """Command line arguments for latent statistics computation."""

    dataset_path: Path
    """Root directory of the dataset (contains metadata.csv, pixel_latents/, voxel_latents/)."""

    num_samples: int | None = None
    """Number of training samples to use for computing statistics (uses first N)."""

    verbose: bool = False
    """Display current mean/std estimates and deltas at each iteration."""

    process_pixel: bool = False
    """Compute pixel latent statistics."""

    process_voxel: bool = False
    """Compute voxel latent statistics."""


if __name__ == "__main__":
    args = tyro.cli(Arguments)

    assert int(args.process_pixel) + int(args.process_voxel) > 0, (
        "At least one of --process_pixel or --process_voxel must be specified"
    )

    compute_latent_statistics(
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        verbose=args.verbose,
        process_pixel=args.process_pixel,
        process_voxel=args.process_voxel,
    )
