# fmt:off
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import tyro
from loguru import logger
from tqdm import tqdm

from utils import str2dict


def collect_hash_identifiers(output_dir):
    paths = []
    local_paths = []
    with tqdm(desc="Enumerating") as pbar:
        for envdataset_dir in (output_dir / 'raw').iterdir():
            for instance_dir in envdataset_dir.iterdir():
                paths.append(instance_dir)
                local_paths.append("/".join(instance_dir.parts[-3:]))
                pbar.update()

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(paths))) as executor, \
            tqdm(total=len(paths), desc="Extracting") as pbar:
        
        def worker(path: Path) -> str | None:
            sha256 = None
            try:
                sha256_path = path / 'sha256.txt'
                assert sha256_path.exists(), f"SHA256 file not found for {sha256_path}"
                with open(sha256_path, 'r') as f:
                    sha256 = f.read().strip()
            except Exception as e:
                logger.error(f"Error extracting for {path}: {e}")
            finally:
                pbar.update()
                return sha256

        sha256s = executor.map(worker, paths)
        executor.shutdown(wait=True)

    metadata = pd.DataFrame(zip(sha256s, local_paths), columns=['sha256', 'local_path'])
    metadata = metadata.set_index('sha256')
    if len(metadata[metadata.index.duplicated()]) > 0:
        logger.warning("Duplicate SHA256 identifiers found. They are most likely generated from an empty data.npz and should be removed.")
        duplicated_sha256 = set(metadata[metadata.index.duplicated()].index.tolist())
        for sha256 in duplicated_sha256:
            local_paths = metadata.loc[sha256]['local_path'].values
            for local_path in local_paths:
                full_path = output_dir / local_path
                if args.delete_duplicated_sha256:
                    if full_path.exists():
                        logger.info(f"Deleting duplicate sha256 at {full_path}")
                        shutil.rmtree(full_path)
                    else:
                        logger.info(f"Path {full_path} does not exist, cannot delete.")
                else:
                    logger.info(f"Duplicate sha256 found at {full_path}. Consider deleting it.")
        metadata = metadata[~metadata.index.duplicated()]
    return metadata

def need_process(key):
    return key in args.field or args.field == ['all']

# merge mt_voxel_classdict_rank.json
def merge_mt_voxel_classdicts(output_dir, timestamp):

    def uniq(lst):
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item
    json_files = list(output_dir.glob('mt_voxel_classdict_*.json'))
    if len(json_files) == 0:
        return
    merged_mt_voxel_classdict = {'node_classes': []}
    for f in json_files:
        with open(f, 'r') as file:
            mt_voxel_classdict = json.load(file)
            merged_mt_voxel_classdict['node_classes'].extend(mt_voxel_classdict['node_classes'])
    merged_mt_voxel_classdict['node_classes'] = list(uniq(sorted(merged_mt_voxel_classdict['node_classes'])))
    new_merged_mt_voxel_classdict = {'node_classes': {},}
    for i, node_class in enumerate(merged_mt_voxel_classdict['node_classes']):
        new_merged_mt_voxel_classdict['node_classes'][i] = node_class

    if (output_dir / 'mt_voxel_classdict.json').exists() and not args.overwrite_mt_voxel_classdict:
        existing_mt_voxel_classdict = json.load(open(output_dir / 'mt_voxel_classdict.json'))
        new_merged_mt_voxel_classdict['node_classes'].update(existing_mt_voxel_classdict['node_classes'])

    for f in json_files:
        shutil.move(f, output_dir / 'merged_records' / f'{timestamp}_{f.name}')

    with open(output_dir / 'mt_voxel_classdict.json', 'w') as f:
        json.dump(new_merged_mt_voxel_classdict, f)

def generate_preprocessing_statistics(metadata, output_dir):
    """Generate statistics about preprocessing conditions from metadata."""
    # Initialize statistics tracking
    metadata_preprocesed = metadata[metadata['preprocessed']]
    total_levels = len(metadata_preprocesed)

    if total_levels == 0:
        return

    condition_counts = {
        'early_termination': 0,
        'moving_player': 0,
        'corrupted_data': 0,
        'framerate_drop': 0,
        'frozen_voxel': 0,
    }
    
    # Count occurrences of each condition
    for _, row in metadata_preprocesed.iterrows():
        if 'level_validity_conditions' in row and row['level_validity_conditions']:
            # Parse the string representation of the conditions
            conditions_str = row['level_validity_conditions']
            try:
                # Handle different possible string formats
                if isinstance(conditions_str, str):
                    # Try to evaluate the string as a Python dictionary
                    conditions = str2dict(conditions_str)
                else:
                    # If it's already a dictionary, use it directly
                    conditions = conditions_str
                
                # Count the conditions
                for condition, triggered in conditions.items():
                    if triggered:
                        condition_counts[condition] += 1
            except Exception as e:
                logger.info(f"Error parsing conditions: {e}")
                continue
    
    # Save statistics to file
    with open(output_dir / 'preprocessing_statistics.txt', 'w') as f:
        f.write(f"Preprocessing Statistics (Total Preprocessed Levels: {total_levels})\n")
        
        valid_count = (metadata['preprocessed'] * metadata['level_quality_score']>0).sum()
        valid_percentage = (valid_count / total_levels) * 100
        f.write(f"\nValid levels: {valid_count} ({valid_percentage:.2f}%)\n")
        f.write(f"Invalid levels: {total_levels - valid_count} ({100 - valid_percentage:.2f}%)\n")

        for condition, count in condition_counts.items():
            percentage = (count / total_levels) * 100
            f.write(f"{condition}: {count} levels ({percentage:.2f}%)\n")

    # also print file contents to terminal
    with open(output_dir / 'preprocessing_statistics.txt', 'r') as f:
        logger.info(f.read())
        logger.info("=" * 50 + "\n")


@dataclass
class Args: # Args(DatasetArgs) is also possible if we want Dataset specific arguments in the future
    """Command line arguments for the program."""
    output_dir: str
    """Directory to save the metadata"""
    field: str = "all"
    """Fields to process, separated by commas"""
    overwrite_metadata: bool = False
    """Overwrite the metadata.csv file. If true, the script will generate a new metadata.csv file from scratch."""
    overwrite_mt_voxel_classdict: bool = False
    """Overwrite the mt_voxel_classdict.json file. If false, the script merges the existing mt_voxel_classdict.json with the new ones."""
    delete_duplicated_sha256: bool = True
    """Delete duplicated sha256 entries found in the dataset."""
    delete_invalid_preprocessing_records: bool = True
    """Delete invalid preprocessing records found in the dataset."""
    update_level_scores_only: bool = False
    """Only update level quality scores from preprocessing records, without changing other metadata fields."""
    from_file: bool = False
    """Build metadata from file instead of from records of processings.
    Useful when some processing fail to generate records but file already exists.
    Note: only works if intermediate files (e.g. voxel_mt_temp folder) have not been deleted."""
    set_validation_samples: Literal["vae", "diffusion", "none"] = "none"
    """If set_validation_samples is True, the script will mark the last N preprocessed levels as validation samples (according to validation_samples_fraction)."""
    validation_samples_fraction: float = 0.05
    """Fraction of levels to mark as validation samples."""
    debug: bool = False
    """Enable debug mode. Will run the code in a single process."""

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.debug:
        from utils import MockThreadPoolExecutor as ThreadPoolExecutor
    else:
        from concurrent.futures import ThreadPoolExecutor

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "merged_records").mkdir(exist_ok=True)

    args.field = args.field.split(",")

    timestamp = str(int(time.time()))

    # get file list
    if (output_dir / "metadata.csv").exists() and not args.overwrite_metadata:
        logger.info("Loading previous metadata...")
        metadata = pd.read_csv(output_dir / "metadata.csv")
        metadata.set_index("sha256", inplace=True)
    else:
        logger.info("Building new metadata...")
        metadata = collect_hash_identifiers(output_dir)

    logger.info("\n" + "=" * 50 + "\n")
    if "preprocessed" not in metadata.columns:
        metadata["preprocessed"] = False
    if "level_quality_score" not in metadata.columns:
        metadata["level_quality_score"] = 0
    if "level_validity_conditions" not in metadata.columns:
        metadata["level_validity_conditions"] = "{}"
    if "voxelized" not in metadata.columns:
        metadata["voxelized"] = False
    if "validation_set" not in metadata.columns:
        metadata["validation_set"] = False
    if "pixel_latent_generated" not in metadata.columns:
        metadata["pixel_latent_generated"] = False
    if "voxel_latent_generated" not in metadata.columns:
        metadata["voxel_latent_generated"] = False

    # Merge preprocessed
    df_files = list(output_dir.glob("preprocessed_*.csv"))
    df_parts = []
    for df_file in df_files:
        try:
            df_i = pd.read_csv(df_file)
            # check for NaN entries and flag them
            if df_i["sha256"].isnull().any():
                if args.delete_invalid_preprocessing_records:
                    logger.warning(f"Deleting incomplete preprocessing record file {df_file}.")
                    os.remove(df_file)
                    mt_vox_classdict_i = df_file.with_name(
                        df_file.name.replace("preprocessed_", "mt_voxel_classdict_").replace(
                            ".csv", ".json"
                        )
                    )
                    if mt_vox_classdict_i.exists():
                        logger.warning(
                            f"Deleting incomplete voxel_class_dict record file {mt_vox_classdict_i}."
                        )
                        os.remove(mt_vox_classdict_i)
                else:
                    logger.warning(f"Skipping file {df_file} as it contains empty records.")
            df_parts.append(df_i)
        except Exception as e:
            logger.warning(f"Skipping file {df_file} raising error {e}.")
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        if args.update_level_scores_only:
            metadata.loc[df.index, "level_quality_score"] = df["level_quality_score"]
        else:
            metadata.update(df, overwrite=True)
        for df_file in df_files:
            shutil.move(df_file, output_dir / "merged_records" / f"{timestamp}_{df_file.name}")

    # Generate preprocessing statistics
    generate_preprocessing_statistics(metadata, output_dir)
    merge_mt_voxel_classdicts(output_dir, timestamp)

    # Merge voxelized
    df_files = list(output_dir.glob("voxelized_*.csv"))
    df_parts = []
    for df_file in df_files:
        try:
            df_parts.append(pd.read_csv(df_file))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        metadata.update(df, overwrite=True)
        for df_file in df_files:
            shutil.move(df_file, output_dir / "merged_records" / f"{timestamp}_{df_file.name}")

    # Merge pixel latents
    df_files = list(output_dir.glob("processed_pixel_latents_*.csv"))
    df_parts = []
    for df_file in df_files:
        try:
            df_parts.append(pd.read_csv(df_file))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        metadata.update(df, overwrite=True)
        for df_file in df_files:
            shutil.move(df_file, output_dir / "merged_records" / f"{timestamp}_{df_file.name}")

    # Merge voxel latents
    df_files = list(output_dir.glob("processed_voxel_latents_*.csv"))
    df_parts = []
    for df_file in df_files:
        try:
            df_parts.append(pd.read_csv(df_file))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        metadata.update(df, overwrite=True)
        for df_file in df_files:
            shutil.move(df_file, output_dir / "merged_records" / f"{timestamp}_{df_file.name}")

    # Set validation samples if requested
    if args.set_validation_samples != "none":
        # Get all ready samples
        if args.set_validation_samples == "vae":
            ready_samples = metadata[metadata["voxelized"]].copy()
        elif args.set_validation_samples == "diffusion":
            ready_samples = metadata[metadata["voxel_latent_generated"] & metadata["pixel_latent_generated"]].copy()
        else:
            raise ValueError(f"Unknown set_validation_samples option: {args.set_validation_samples}")
        ready_samples = ready_samples[ready_samples["level_quality_score"] > 0]
        if len(ready_samples) > 0:
            # Calculate number of validation samples
            num_validation = max(1, int(len(ready_samples) * args.validation_samples_fraction))
            # Mark the last N samples as validation samples
            validation_indices = ready_samples.index[-num_validation:]
            metadata.loc[validation_indices, "validation_set"] = True

    # build metadata from files
    if args.from_file:
        if need_process("preprocessed"):
            logger.warning("--from_file not fully supported for preprocessed as no files are produced. "
                           "Instead we check whether a file exists in voxel_classes/. "
                           "Level_validity_conditions information will be lost.")
        with (
            ThreadPoolExecutor(max_workers=os.cpu_count()) as executor,
            tqdm(total=len(metadata), desc="Building metadata") as pbar,
        ):

            def worker(sha256):
                try:
                    if need_process("preprocessed"):
                        if (output_dir / "voxel_classes" / f"{sha256}.npz").exists():
                            metadata.loc[sha256, "preprocessed"] = True
                            metadata.loc[sha256, "level_quality_score"] = 10
                        else:
                            metadata.loc[sha256, "preprocessed"] = False
                            metadata.loc[sha256, "level_quality_score"] = 0
                        # try:
                        #     metadata.loc[sha256]
                        #     # previously we check mt_tmp but now we don't save this file
                        #     metadata.loc[sha256, "preprocessed"] = True
                        #     assert metadata.loc[sha256, "level_quality_score"] >= 0
                        # except Exception as e:
                        #     metadata.loc[sha256, "preprocessed"] = False
                        #     metadata.loc[sha256, "level_quality_score"] = 0
                    if need_process("voxelized"):
                        if (output_dir / "voxel_classes" / f"{sha256}.npz").exists():
                            metadata.loc[sha256, "voxelized"] = True
                        else:
                            metadata.loc[sha256, "voxelized"] = False

                    # Check for pixel latent files
                    pixel_latents_dir = output_dir / "pixel_latents"
                    if pixel_latents_dir.exists():
                        latent_path = pixel_latents_dir / f"{sha256}.safetensors"
                        if not latent_path.exists():
                            latent_path = pixel_latents_dir / f"{sha256}.npz"
                        metadata.loc[sha256, "pixel_latent_generated"] = latent_path.exists()

                    # Check for voxel latent files
                    voxel_latents_dir = output_dir / "voxel_latents"
                    if voxel_latents_dir.exists():
                        latent_path = pixel_latents_dir / f"{sha256}.safetensors"
                        if not latent_path.exists():
                            latent_path = pixel_latents_dir / f"{sha256}.npz"
                        metadata.loc[sha256, "voxel_latent_generated"] = latent_path.exists()

                    pbar.update()
                except Exception as e:
                    logger.info(f"Error processing {sha256}: {e}")
                    pbar.update()

            executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    # statistics
    metadata.to_csv(output_dir / "metadata.csv", index=(metadata.index.name == "sha256"))
    num_generated = metadata["local_path"].count() if "local_path" in metadata.columns else 0
    with open(output_dir / "statistics.txt", "w") as f:
        f.write("Statistics:\n")
        f.write(f"  - Number of assets: {len(metadata)}\n")
        f.write(f"  - Number of assets generated: {num_generated}\n")
        f.write(f"  - Number of assets preprocessed: {metadata['preprocessed'].sum()}\n")
        f.write(f"  - Number of assets voxelized: {metadata['voxelized'].sum()}\n")
        f.write(f"  - Number of assets with pixel latents: {metadata['pixel_latent_generated'].sum()}\n")
        f.write(f"  - Number of assets with voxel latents: {metadata['voxel_latent_generated'].sum()}\n")
        if args.set_validation_samples:
            f.write(f"  - Number of validation samples: {metadata['validation_set'].sum()}\n")

    with open(output_dir / "statistics.txt", "r") as f:
        logger.info(f.read())
        logger.info("=" * 50 + "\n")
