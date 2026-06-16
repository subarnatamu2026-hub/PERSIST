import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration
from easydict import EasyDict as edict
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader

from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraAction
from models import camera_transformer
from trainers import BaseTrainer, BaseTrainingArgs
from utils.camera_util import angular_diff_deg, camera_to_xyz_pry_fov

warnings.simplefilter(action="ignore", category=FutureWarning)

@dataclass(kw_only=True)
class TrainingArgs(BaseTrainingArgs):
    """Configuration for training the model."""

    model: camera_transformer.CameraTransformerArgs
    """Camera prediction model configuration."""

    run_name: str = "debug-transformer-camera"
    """Experiment name"""

    output_dir: str = "outputs/runs/camera_transformer"
    """Output directory"""

    camera_offset_ts: int = 1
    """Time offset to apply to camera conditioning"""

    voxel_offset_ts: int = 0
    """Time offset to apply to voxel conditioning"""

    remove_bad_frames: bool = True
    """Whether to filter out frames not satisfying delay/inconsistency heuristics."""

    random_seed: int = 42

    remote_output_dir: str | None = None
    """For Azure only. Remote directory for saving the output."""

    ckpt_to_load: str | None = None
    """Path to a specific accelerator checkpoint folder which we want to resume from"""

    dataset_path: str = "datasets"
    """Path to training dataset"""

    min_level_quality_score: float = 1.0
    """Levels below this quality score will be filtered out from the dataset"""

    batch_size: int = 2
    """Batch size for training (per device)"""

    grad_accumulation_steps: int = 1
    """Number of gradient accumulation steps"""

    total_steps: int | None = None
    """Total training steps to run. total_grad_steps = total_steps // grad_accumulation_steps.
    Setting both total_steps and total_grad_steps will throw an error."""

    total_grad_steps: int | None = None
    """Total gradient steps to run. 
    Setting both total_steps and total_grad_steps will throw an error."""

    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"

    log_train_metrics: bool = True
    """Whether to log extra training metrics (binned losses by diffusion time etc)."""

    loss_coef_xyz: float = 100.0

    loss_coef_rotation: float = 0.1

    loss_coef_fov: float = 1.0

    xyz_rescaling: float = 48.0

    max_grad_norm: float = 1.0

    learning_rate: float = 1e-04

    lr_warmup_steps: int = 10_000
    """Number of steps to warm up the learning rate. Warm up finishes at lr_warmup_steps * grad_accumulation_steps"""

    weight_decay: float = 2e-3

    opt_beta: Tuple[float, float] = (0.9, 0.99)

    # Frequency

    i_print: int = 100
    """Print every i grad_step."""

    i_log: int = 100
    """Log every i grad_step."""

    i_val: int = 100
    """Run validation every i grad_step. Set to 0 to disable validation during training."""

    i_save: int = 10_000
    """Save every i grad_step. Set to 0 to disable regular checkpoint saving during training."""

    def __post_init__(self):
        super().__post_init__()
        assert self.total_steps or self.total_grad_steps, "Either total_steps or total_grad_steps must be set."
        assert not (self.total_steps and self.total_grad_steps), "Only one of total_steps or total_grad_steps can be set."
        if self.total_grad_steps:
            self.total_steps = self.total_grad_steps * self.grad_accumulation_steps
        else:
            self.total_grad_steps = self.total_steps // self.grad_accumulation_steps
        if self.i_val:
            assert self.i_log % self.i_val == 0, "i_log must be a multiple of i_val"

class VoxelDiffuserTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)

    def prepare_dataset(self):
        logger.info("Build Datasets ...")
        self.dataset = MinetestLatentCameraAction(
            self.cfg.dataset_path,
            min_level_quality_score=args.min_level_quality_score,
            clip_len=args.model.context_window_size + 1,
            no_latents=True,
            sample_cropped_voxel=True,
            normalize_latents=False,
            voxel_offset_ts=args.voxel_offset_ts,
            camera_offset_ts=args.camera_offset_ts,
            cam_rotation_representation=args.model.cam_rotation_input_rep,
            cam_translation_representation=args.model.cam_translation_input_rep,
            sample_voxel_importance_mask=args.remove_bad_frames,
            remove_bad_frames=args.remove_bad_frames,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataloader_workers,
            persistent_workers=True if self.cfg.dataloader_workers else False,
            prefetch_factor=self.cfg.dataloader_prefetch_factor if self.cfg.dataloader_workers else None,
            pin_memory=True,
            drop_last=True,
            collate_fn=getattr(self.dataset, "collate_fn", None),
        )

        self.val_dataset = MinetestLatentCameraAction(
            self.cfg.dataset_path,
            min_level_quality_score=args.min_level_quality_score,
            clip_len=args.model.context_window_size + 1,
            split="val",
            no_latents=True,
            sample_cropped_voxel=True,
            normalize_latents=False,
            voxel_offset_ts=args.voxel_offset_ts,
            camera_offset_ts=args.camera_offset_ts,
            cam_rotation_representation=args.model.cam_rotation_input_rep,
            cam_translation_representation=args.model.cam_translation_input_rep,
            sample_voxel_importance_mask=args.remove_bad_frames,
            remove_bad_frames=args.remove_bad_frames,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.val_dataloader_workers,
            persistent_workers=True if self.cfg.val_dataloader_workers else False,
            prefetch_factor=self.cfg.val_dataloader_prefetch_factor if self.cfg.val_dataloader_workers else None,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

    # fmt:off
    def prepare_models(self):
        logger.info("Build models ...")
        classdict_path = os.path.join(self.dataset.root, 'mt_voxel_classdict.json')
        with open(classdict_path, 'r') as f:
            voxel_classdict = json.load(f)
        vocab_size = len(voxel_classdict['node_classes'])
        if vocab_size != self.cfg.model.voxel_vocab_size:
            raise ValueError(f"Camera model voxel vocab size {self.cfg.model.voxel_vocab_size} does not match voxel "
                             f"class vocab size {vocab_size} of the training dataset.")

        self.models = nn.ModuleDict(
            {
                "camera": getattr(camera_transformer, self.cfg.model.name)(**vars(self.cfg.model)
                ),
            }
        )
        num_parameters = sum(p.numel() for p in self.models.parameters())
        logger.info(f"# of parameters: {num_parameters / 1e6: .4f}M")

    def loss_delta_camera(self, batch, model_cfg):
        assert model_cfg.cam_rotation_input_rep == "6d" # only support 6d input for now
        assert model_cfg.cam_translation_input_rep == "voxel_local"

        assert model_cfg.cam_rotation_output_rep in ["delta_angle", "delta_angle_cat"]
        assert model_cfg.cam_translation_output_rep in ["voxel_local", "delta_voxel_local"]
        assert model_cfg.cam_fov_output_rep == "delta"

        # 1. compute targets
        with torch.no_grad():
            xyz, pry, fov = camera_to_xyz_pry_fov(
                batch["camera"], in_degrees=True, rotation_rep=model_cfg.cam_rotation_input_rep)

            if model_cfg.cam_translation_output_rep == "voxel_local":
                target_xyz = xyz[:, 1:]
            elif model_cfg.cam_translation_output_rep == "delta_voxel_local":
                target_xyz = xyz[:, 1:] - xyz[:, :-1]
            target_fov = fov[:, 1:] - fov[:, :-1] #deg/frame
            target_fov = target_fov.unsqueeze(-1)

            # rotation targets
            delta_pry = pry[:, 1:] - pry[:, :-1]
            delta_pry[..., 2] = angular_diff_deg(pry[:, 1:, 2], pry[:, :-1, 2])
            delta_py = torch.stack([delta_pry[:, :, 0], delta_pry[:, :, 2]], dim=-1)
            if model_cfg.cam_rotation_output_rep == "delta_angle":
                target_py = delta_py
            elif model_cfg.cam_rotation_output_rep == "delta_angle_cat":
                #for pitch and yaw, assign labels [0, 1, 2] for [negative, zero, positive]
                target_py = torch.zeros_like(delta_py, dtype=torch.long)
                target_py[(delta_py >= -1e-3) & (delta_py <= 1e-3)] = 1
                target_py[delta_py > 1e-3] = 2

        # 2. obtain predictions
        ctx_len = self.cfg.model.context_window_size
        x = batch["camera"][:, :ctx_len]
        # build conditioning
        action = batch["action"][:, :ctx_len]
        voxel = batch["cropped_voxel"][:, :ctx_len]
        # forward pass
        pred = self.models['camera'](x, action, voxel)

        #3. compute loss
        terms = edict(loss=0.0, loss_rot=0.0, loss_xyz=0.0, loss_fov=0.0)

        if model_cfg.cam_rotation_output_rep == "delta_angle":
            pred_py, pred_xyz, pred_fov = pred.split([2, 3, 1], dim=-1)
            terms.loss_rot = F.mse_loss(pred_py, target_py)
        elif model_cfg.cam_rotation_output_rep == "delta_angle_cat":
            pred_py, pred_xyz, pred_fov = pred.split([6, 3, 1], dim=-1)
            pred_py = rearrange(pred_py, 'b t (r c) -> (b t) r c', r=2, c=3)
            target_py = rearrange(target_py, 'b t r -> (b t) r')
            loss_p = F.cross_entropy(pred_py[..., 0, :], target_py[..., 0], reduction="mean")
            loss_y = F.cross_entropy(pred_py[..., 1, :], target_py[..., 1], reduction="mean")
            terms.loss_rot = loss_p + loss_y

        terms.loss_xyz = F.mse_loss(pred_xyz, target_xyz)
        terms.loss_fov = F.mse_loss(pred_fov, target_fov)
        terms.loss = (
                self.cfg.loss_coef_rotation * terms.loss_rot +
                self.cfg.loss_coef_xyz * terms.loss_xyz +
                self.cfg.loss_coef_fov * terms.loss_fov
        )

        return terms

    def loss_camera(self, batch):
        ctx_len = self.cfg.model.context_window_size
        x = batch["camera"][:, :ctx_len]
        target = batch["camera"][:, 1:].clone()
        target[..., -4:-1] = target[..., -4:-1] * self.cfg.xyz_rescaling  # rescale xyz

        # build conditioning
        action = batch["action"][:, :ctx_len]
        voxel = batch["cropped_voxel"][:, :ctx_len]

        pred = self.models['camera'](x, action, voxel)
        pred[..., -4:-1] = pred[..., -4:-1] * self.cfg.xyz_rescaling  # rescale xyz
        terms = edict(loss=0.0)
        terms.loss = F.mse_loss(pred, target)

        return terms

    def training_losses(
            self,
            batch,
            verbose=True,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            batch: the batch of data

        Returns:
            a dict with the key "loss" containing a tensor of shape [B].
            may also contain other keys for different terms.
        """

        model_cfg = self.cfg.model
        if model_cfg.cam_translation_input_rep == model_cfg.cam_translation_output_rep and model_cfg.cam_rotation_input_rep == model_cfg.cam_rotation_output_rep:
            terms = self.loss_camera(batch)
        else:
            terms = self.loss_delta_camera(batch, model_cfg)

        return terms, {}


def main(args):
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        automatic_checkpoint_naming=False,
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        project_config=project_config,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=10))],
        # step_scheduler_with_optimizer=False # uncomment this if you want to use a scheduler
    )
    trainer = VoxelDiffuserTrainer(args, accelerator)
    trainer.prepare_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_optimizer()
    trainer.prepare_accelerate()
    trainer.train()


if __name__ == "__main__":
    # Prepare config
    args = tyro.cli(TrainingArgs)
    main(args)
