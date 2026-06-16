import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import *

import numpy as np
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
from models import dit_voxel
from trainers.flow_euler import FlowEulerTrainingArgs, FlowEuler
from trainers import BaseTrainer

warnings.simplefilter(action="ignore", category=FutureWarning)


def fast_psnr(mse, data_range, base=10.0):
    """Compute PSNR from MSE and data range."""
    psnr_base_e = 2 * torch.log(torch.tensor(data_range)) - torch.log(torch.maximum(mse, torch.tensor(1e-10)))
    psnr = psnr_base_e * (10 / torch.log(torch.tensor(base)))
    return psnr


@dataclass(kw_only=True)
class TrainingArgs(FlowEulerTrainingArgs):
    """Configuration for training the model."""

    denoiser: dit_voxel.VoxelDitDenoiserArgs
    """Denoiser model to use for diffusion."""

    run_name: str = "debug-diffuser-voxel"
    """Experiment name"""

    output_dir: str = "outputs/runs/voxel_diffuser"
    """Output directory"""

    loss_target: Literal["x0", "v"] = "v"
    """Loss target for diffusion model"""

    camera_offset_ts: int = 1
    """Time offset to apply to camera conditioning"""

    camera_translation_representation: Literal["extrinsics", "voxel_local"] = "voxel_local"

    camera_rotation_representation: Literal["quaternion", "6d", "cam_dir"] = '6d'

    remove_bad_frames: bool = True
    """Whether to filter out frames not satisfying delay/inconsistency heuristics."""

    def __post_init__(self):
        super().__post_init__()
        #raymap currently only supports voxel_local
        if self.denoiser.pixel_use_raymap:
            assert self.camera_translation_representation == "voxel_local"


class VoxelDiffuserTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)
        self.flow_euler = FlowEuler(cfg.diffusion, device=accelerator.device)

    def prepare_dataset(self):
        logger.info("Build Datasets ...")
        self.dataset = MinetestLatentCameraAction(
            self.cfg.dataset_path,
            min_level_quality_score=args.min_level_quality_score,
            clip_len=args.denoiser.context_window_size,
            normalize_latents=args.normalize_latents,
            pixel_offset_ts=1,
            camera_offset_ts=args.camera_offset_ts,
            cam_rotation_representation=args.camera_rotation_representation,
            cam_translation_representation=args.camera_translation_representation,
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
            min_level_quality_score=self.cfg.min_level_quality_score,
            clip_len=args.denoiser.context_window_size,
            split="val",
            normalize_latents=args.normalize_latents,
            pixel_offset_ts=1,
            camera_offset_ts=args.camera_offset_ts,
            cam_rotation_representation=args.camera_rotation_representation,
            cam_translation_representation=args.camera_translation_representation,
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
        self.models = nn.ModuleDict(
            {
                "denoiser": getattr(dit_voxel, self.cfg.denoiser.name)(**vars(self.cfg.denoiser)
                ),
            }
        )
        num_parameters = sum(p.numel() for p in self.models.parameters())
        logger.info(f"# of parameters: {num_parameters / 1e6: .4f}M")

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
        x_0 = batch["voxel"]

        # build conditioning
        cond = {
            "action": batch["action"].clone(),
            "camera": batch["camera"].clone(),
        }
        # noise augmentation on pixel latents
        noise_pix = torch.randn_like(batch["pixel"])
        noise_pix = torch.clamp(noise_pix, -self.flow_euler.cfg.noise_abs_max, self.flow_euler.cfg.noise_abs_max)
        cond["pixel_latents"] = (1 - self.flow_euler.cfg.noise_cond) * batch["pixel"] + self.flow_euler.cfg.noise_cond * noise_pix

        noise = torch.randn_like(x_0)
        noise = torch.clamp(noise, -self.flow_euler.cfg.noise_abs_max, self.flow_euler.cfg.noise_abs_max)
        t = self.flow_euler.sample_t(*x_0.shape[:2]).to(x_0.device).float()
        x_t = self.flow_euler.diffuse(x_0, t, noise)

        pred = self.models['denoiser'](x_t, t, cond)
        if self.cfg.loss_target == "x0":
            target = x_0
        elif self.cfg.loss_target == "v":
            target = self.flow_euler.get_v_from_noise(x_0, noise)
        else:
            raise ValueError(f"Unknown loss target: {self.cfg.loss_target}")

        terms = edict(loss=0.0)
        terms.loss = F.mse_loss(pred, target)
        if verbose:
            with torch.no_grad():
                loss_per_timestep = F.mse_loss(
                    pred, target, reduction="none").mean(dim=list(range(2, len(pred.shape))))

        if verbose:
            with torch.no_grad():
                if self.cfg.loss_target == "x0":
                    loss_data_space = F.mse_loss(pred, x_0).mean()
                elif self.cfg.loss_target == "v":
                    pred_x_0= self.flow_euler.sample_x0(x_t, t, pred)
                    loss_data_space = F.mse_loss(pred_x_0, x_0).mean()
                terms["z_psnr"] = fast_psnr(loss_data_space, data_range=2*self.flow_euler.cfg.noise_abs_max)
                time_bin = np.digitize(t.view(-1).cpu().numpy(), np.linspace(0, 1, 11)) - 1
                for i in range(10):
                    if (time_bin == i).sum() != 0:
                        terms[f"bin_{i}"] = loss_per_timestep.view(-1)[time_bin == i].mean()
                if "action" and "camera" in cond:
                    tt = rearrange(t, "B T -> (B T)")
                    t_emb = self.get_model(self.models['denoiser']).t_embedder(tt * self.cfg.denoiser.timestep_scaling_factor).abs()
                    t_emb = rearrange(t_emb, "(B T) D -> B T D", B=t.shape[0], T=t.shape[1])
                    camera_for_ada = cond["camera"].clone()
                    if self.get_model(self.models["denoiser"]).camera_use_ape:
                        camera_for_ada = rearrange(camera_for_ada, "b t d -> (b t) d")
                        cam_rot, cam_xyz, cam_fov = camera_for_ada.split([6, 3, 1], dim=-1)
                        cam_rot = self.get_model(self.models["denoiser"]).cam_embedding_layer_rot(cam_rot)
                        cam_xyz = self.get_model(self.models["denoiser"]).cam_embedding_layer_xyz(cam_xyz)
                        camera_for_ada = torch.cat((cam_rot, cam_xyz, cam_fov), dim=-1)
                        camera_for_ada = rearrange(camera_for_ada, "(b t) d -> b t d", b=t.shape[0], t=t.shape[1])
                    else:
                        camera_for_ada = camera_for_ada * self.get_model(self.models["denoiser"]).camera_pos_scaling_factor
                    if self.get_model(self.models["denoiser"]).action_camera_embedder:
                        act_emb = self.get_model(self.models["denoiser"]).action_camera_embedder(cond["action"], camera_for_ada).abs()
                        cam_emb = act_emb
                        c_emb = act_emb + t_emb
                    else:
                        act_emb = self.get_model(self.models["denoiser"]).action_embedder(cond["action"]).abs()
                        cam_emb = self.get_model(self.models["denoiser"]).camera_embedder(camera_for_ada).abs()
                        c_emb = act_emb + cam_emb + t_emb
                    terms["t_emb_frac"] = (t_emb / c_emb).mean()
                    terms["act_emb_frac"] = (act_emb / c_emb).mean()
                    terms["cam_emb_frac"] = (cam_emb / c_emb).mean()
                    terms["max_cam_pos"] = cond["camera"][..., -4:-1].abs().max()

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
        # step_scheduler_with_optimizer=False # uncomment this if you want to use an lr scheduler
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
