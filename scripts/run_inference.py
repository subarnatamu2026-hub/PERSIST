"""Run the voxel world-model pipeline autoregressively and save a video.

There is a single pipeline folder, `pipelines/persist_S_pipeline`. It exposes two
variants selected with `--pipeline-variant`: `S` (default) and `XL` (voxel-denoiser
patch_size 1, sampler noise_cond 0.0). Model weights are never stored next to the
pipeline folder; supply them either from HuggingFace (`--checkpoints-namespace`) or as
explicit local paths (`--checkpoints.*`). Likewise, the initial-context dataset comes
from either a local folder (`--dataset-path`) or HuggingFace (`--dataset-repo`).

The initial context (first pixel frame, initial voxel grid, camera, action sequence)
is sampled from a dataset via `MinetestLatentCameraActionEval`, so all of the data
loading / transforms / camera representation are handled by the data loader. The
camera is always taken from the data; there is no built-in default.

Examples:
    # Download everything from HuggingFace (no local files needed)
    python -m scripts.run_inference \
        --pipeline-variant S \
        --checkpoints-namespace PERSIST-team \
        --dataset-repo PERSIST-team/persist-eval-sample \
        --num-frames 16 \
        --output-dir outputs/rollouts

    # Local weights + local dataset
    python -m scripts.run_inference \
        --pipeline-variant S \
        --dataset-path datasets/example \
        --num-frames 16 \
        --checkpoints.voxel-encoder ckpts/voxel_vae/model.safetensors \
        --checkpoints.voxel-decoder ckpts/voxel_vae/model_1.safetensors \
        --checkpoints.pixel-vae ckpts/pixel_vae/model.safetensors \
        --checkpoints.camera-model ckpts/camera_model/model.safetensors \
        --checkpoints.voxel-denoiser ckpts/voxel_dit/model.safetensors \
        --checkpoints.pixel-denoiser ckpts/pixel_dit/model.safetensors \
        --checkpoints.voxel-class-embedder ckpts/pixel_dit/model_1.safetensors \
        --output-dir outputs/rollouts

    # XL variant: identical command with --pipeline-variant XL
    # Specific episodes: --eval-instances <sha256>,<sha256>

    # Multi-GPU rollout (episodes sharded across GPUs) with bf16 mixed precision:
    accelerate launch --multi_gpu --num_processes <N_GPU> -m scripts.run_inference \
        --pipeline-variant S \
        --checkpoints-namespace PERSIST-team \
        --dataset-repo PERSIST-team/persist-eval-sample \
        --num-frames 16 --mixed-precision bf16 \
        --output-dir outputs/rollouts
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

# Make the repo root importable whether invoked as `python -m scripts.run_inference`
# or `python scripts/run_inference.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tyro
from accelerate import Accelerator
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.io import write_video

from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraActionEval
from pipelines.pipeline import REQUIRED_MODEL_KEYS, VoxelFirstPipeline, pipeline_variant_overrides


@dataclass
class Checkpoints:
    """Filesystem path to each model's trained weights (one per pipeline model).

    Optional: leave unset to fetch the weights from HuggingFace via
    ``--checkpoints-namespace``. Any path supplied here overrides the
    corresponding file downloaded from the HF namespace for that model.
    The pipeline folder ships configs only, never weights.
    """

    voxel_encoder: Optional[str] = None
    voxel_decoder: Optional[str] = None
    pixel_vae: Optional[str] = None
    camera_model: Optional[str] = None
    voxel_denoiser: Optional[str] = None
    pixel_denoiser: Optional[str] = None
    voxel_class_embedder: Optional[str] = None


@dataclass
class Args:
    """Inference configuration."""

    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    """Local paths to the trained model weights (specify each with --checkpoints.<model>).
    Optional when --checkpoints-namespace is given; any path set here overrides the HF download."""

    checkpoints_namespace: Optional[str] = None
    """HF org/user holding one model repo per checkpoint (e.g. PERSIST-team). Each repo
    <namespace>/<prefix><model-key> must contain a single 'model.safetensors'. When set, weights are
    downloaded for every model key not overridden by an explicit --checkpoints.<model> path."""
    checkpoints_repo_prefix: str = "persist-"
    """Prefix for the per-model HF repo names: repo = <namespace>/<prefix><model-key-with-dashes>."""
    checkpoints_revision: Optional[str] = None
    """Optional git revision/tag/commit applied to all weight repos."""

    dataset_path: Optional[str] = None
    """Local dataset to sample the initial context from (built with the standard dataset toolkits).
    Provide exactly one of --dataset-path or --dataset-repo."""
    dataset_repo: Optional[str] = None
    """HF dataset repo id to sample the initial context from (e.g. PERSIST-team/persist-eval-sample).
    Downloaded via snapshot_download(repo_type='dataset'). Provide exactly one of --dataset-path or --dataset-repo."""
    dataset_revision: Optional[str] = None
    """Optional git revision/tag/commit for --dataset-repo."""

    pipeline_variant: Literal["S", "XL"] = "S"
    """Pipeline variant. 'S' is the base config; 'XL' uses voxel-denoiser patch_size 1 and noise_cond 0.0."""
    pipeline_path: str = "pipelines/persist_S_pipeline"
    """Pipeline config folder (pipeline.json + model_configs.py); also accepts an HF repo id."""
    compile_transformer_blocks: bool = False
    """Whether to compile the transformer blocks for the pixel/voxel denoiser for faster inference. Only enable if your 
    environment and GPU supports torch.compile"""

    eval_instances: Optional[str] = None
    """Comma-separated sha256 hashes to roll out. Defaults to the dataset's validation split."""
    num_instances: Optional[int] = None
    """Optional cap on how many episodes to load (after instance/split filtering)."""
    batch_size: int = 1
    """Number of episodes to roll out per-GPU."""

    num_frames: int = 16
    """Number of frames to autoregressively generate."""
    use_camera_gt: bool = False
    """If set, feed the ground-truth camera trajectory instead of predicting it with the camera model."""

    include_initial_voxel_frame: bool = False
    """If set, provide the dataset's initial voxel grid as context. Otherwise the pipeline generates
    the initial voxel frame from the initial pixel frame."""
    include_initial_pixel_frame: bool = False
    """If set, use the dataset's initial pixel frame as the first generated frame (pixel_use_x0).
    Otherwise the initial pixel frame is used only to seed the initial voxel and is then discarded
    (the pipeline regenerates it)."""
    use_kv_cache: bool = True
    """Use a KV cache during the autoregressive rollout (faster). Disable with --no-use-kv-cache."""

    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    """Autocast precision for the rollout (matches `accelerate launch --mixed_precision`)."""

    output_dir: str = "outputs/rollouts"
    fps: int = 24
    device: str = "cuda:0"
    """CUDA device for single-process runs. Under `accelerate launch` the device is managed by
    accelerate (one process per GPU) and this is ignored."""


def resolve_checkpoints(args: Args) -> dict:
    """Return ``{model_key: local_path}`` for every pipeline model.

    Resolution order, per key:
      1. an explicit ``--checkpoints.<model>`` local path (always wins);
      2. ``model.safetensors`` inside the snapshot of ``<namespace>/<prefix><model-key>``.

    The ``voxel_denoiser`` is the only variant-specific checkpoint: it is fetched from
    ``<prefix>voxel-denoiser-<variant>`` (e.g. ``persist-voxel-denoiser-xl``); all other models are
    shared across variants. Backwards compatible: with no namespace and all 7 local paths given,
    this behaves exactly as the previous ``vars(args.checkpoints)``.
    """
    overrides = {k: v for k, v in vars(args.checkpoints).items() if v}
    cfg: dict = {}
    if args.checkpoints_namespace:
        from huggingface_hub import hf_hub_download

        for key in REQUIRED_MODEL_KEYS:
            repo_name = f"{args.checkpoints_repo_prefix}{key.replace('_', '-')}"
            if key == "voxel_denoiser":
                # S and XL ship separately-trained voxel denoisers; everything else is shared.
                repo_name = f"{repo_name}-{args.pipeline_variant.lower()}"
            repo_id = f"{args.checkpoints_namespace}/{repo_name}"
            cfg[key] = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                revision=args.checkpoints_revision,
            )
    cfg.update(overrides)  # explicit local paths win

    missing = [k for k in REQUIRED_MODEL_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(
            f"No checkpoint source for: {missing}. Provide --checkpoints-namespace "
            f"or --checkpoints.<model> for each missing model."
        )
    return cfg


def resolve_dataset_root(args: Args) -> str:
    """Return a local dataset directory, downloading from HF if --dataset-repo is set."""
    if args.dataset_repo and args.dataset_path:
        raise ValueError("Provide exactly one of --dataset-path or --dataset-repo, not both.")
    if args.dataset_repo:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=args.dataset_repo,
            revision=args.dataset_revision,
            repo_type="dataset",
        )
    if args.dataset_path:
        return args.dataset_path
    raise ValueError("Provide exactly one of --dataset-path or --dataset-repo.")


def build_dataloader(args: Args) -> DataLoader:
    root = resolve_dataset_root(args)
    dataset = MinetestLatentCameraActionEval(
        root,
        clip_len=args.num_frames,
        instances=args.eval_instances,
        num_instances=args.num_instances,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No episodes selected from {root} "
            f"(eval_instances={args.eval_instances}). Nothing to roll out."
        )
    logger.info(f"Loaded {len(dataset)} episode(s) from {root}.")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )


def build_context(batch: dict, num_frames: int, use_camera_gt: bool, include_initial_voxel_frame: bool) -> dict:
    """Construct a pipeline context from a sampled batch.

    The eval dataset returns raw pixel frames and voxel classes (no latents); the
    pipeline encodes them. Camera comes straight from the data (voxel-local 6D + xyz + fov).

    The initial pixel frame is always provided. The initial voxel grid is included only when
    `include_initial_voxel_frame` is set; otherwise the pipeline generates the initial voxel from
    the initial pixel frame.
    """
    context = {
        "action": batch["action"][:, :num_frames],
        "pixel": batch["raw_images"][:, :1],        # (B, 1, 3, H, W); pipeline encodes
        # Full trajectory when feeding GT camera; otherwise just the initial frame.
        "camera": batch["camera"] if use_camera_gt else batch["camera"][:, :1],
    }
    if include_initial_voxel_frame:
        context["voxel"] = batch["voxel_classes"][:, :1]  # (B, 1, X, Y, Z) class ids; pipeline encodes
    return context


def main(args: Args):
    # One Accelerator drives both mixed precision (autocast) and multi-GPU. Under
    # `accelerate launch --multi_gpu` each process owns one GPU; without it this is a
    # single-process run and we honour --device.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    distributed = accelerator.num_processes > 1
    device = accelerator.device if distributed else torch.device(args.device)

    model_config_overrides, sampler_args = pipeline_variant_overrides(args.pipeline_variant)
    if accelerator.is_main_process:
        logger.info(f"Loading persist_S pipeline (variant {args.pipeline_variant}) from: {args.pipeline_path}")
        logger.info(f"Processes: {accelerator.num_processes} | device: {device} | mixed_precision: {args.mixed_precision}")
    pipe = VoxelFirstPipeline.from_pretrained(
        args.pipeline_path,
        device=device,
        accelerator=accelerator,
        custom_checkpoint_cfg=resolve_checkpoints(args),
        model_config_overrides=model_config_overrides,
        voxel_sampler_args=sampler_args or None,
        pixel_sampler_args=sampler_args or None,
    )
    if args.compile_transformer_blocks:
        pipe.compile_blocks()

    dataloader = build_dataloader(args)
    instances = dataloader.dataset.instances  # capture before prepare() wraps the loader
    if distributed:
        # Shards batches across processes; even_batches (default) pads so every process runs the
        # same number of pipe() calls, keeping the rollout's periodic wait_for_everyone aligned.
        dataloader = accelerator.prepare(dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    to_uint8 = lambda x: torch.clamp(((x + 1) / 2) * 255, 0, 255).to(torch.uint8)
    for batch in dataloader:
        context = build_context(batch, args.num_frames, args.use_camera_gt, args.include_initial_voxel_frame)
        with accelerator.autocast():
            rollout = pipe(
                context,
                args.num_frames,
                output_latents=False,
                output_decoded_pixels=True,
                pixel_use_x0=args.include_initial_pixel_frame,
                use_camera_gt=args.use_camera_gt,
                keep_on_device=False,
                use_kv_cache=args.use_kv_cache,
                verbose=accelerator.is_main_process,
            )

        if distributed:
            accelerator.wait_for_everyone()

        for b, instance_idx in enumerate(batch["instance_idx"].tolist()):
            instance = instances[instance_idx]
            out_path = os.path.join(args.output_dir, f"{args.pipeline_variant}_{instance}.mp4")
            pixel = rearrange(rollout["pixel"][b], "T C H W -> T H W C")
            write_video(
                out_path,
                to_uint8(pixel).cpu(),
                fps=args.fps,
                video_codec="libx264",
                options={"crf": "18", "preset": "veryfast"},
            )
            logger.info(f"Saved rollout video to {out_path}")

        if distributed:
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    main(tyro.cli(Args))
