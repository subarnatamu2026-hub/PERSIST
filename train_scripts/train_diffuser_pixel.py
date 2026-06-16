# fmt:off
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Literal, Tuple

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
import safetensors.torch

from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraAction
from models import dit_pixel, vae_voxel
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

    denoiser: dit_pixel.FrameDepthStackPixelDitDenoiserArgs
    """Denoiser model to use for diffusion."""

    run_name: str = "debug-diffuser-pixel"
    """Experiment name"""

    output_dir: str = "outputs/runs/pixel_diffuser"
    """Output directory"""

    loss_target: Literal["x0", "v"] = "v"
    """Loss target for diffusion model"""

    loss_type: Literal["mse","ramp_mse"] = "mse"

    voxel_decoder_checkpoint: str | None = None
    """Path to voxel VAE decoder checkpoint. If provided, will decode voxel latents to voxel tensors before passing to model"""

    voxel_decoder: vae_voxel.ResNetDecoderArgs
    """Voxel decoder configuration"""

    voxel_embedding_dim: int = 32
    """Dimension of learned voxel class embeddings (default matches voxel latent channels)"""
    
    voxel_decoder_batch_size: int | None = None
    """Batch size for voxel decoder during training"""

    camera_rotation_representation: Literal["6d", "quaternions"] = "6d"

    remove_bad_frames: bool = True
    """Whether to filter out frames not satisfying delay/inconsistency heuristics."""

    def __post_init__(self):
        super().__post_init__()

        # Adjust denoiser config based on whether voxel decoder is used
        if self.voxel_decoder_checkpoint is not None:
            assert self.denoiser.aggregation_config.num_layers == 4*self.denoiser.voxel_dim

class PixelDiffuserTrainer(BaseTrainer):

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
            cam_rotation_representation=args.camera_rotation_representation,
            cam_translation_representation="extrinsics",
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
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        self.val_dataset = MinetestLatentCameraAction(
            self.cfg.dataset_path,
            min_level_quality_score=self.cfg.min_level_quality_score,
            clip_len=args.denoiser.context_window_size,
            split="val",
            normalize_latents=args.normalize_latents,
            cam_rotation_representation=args.camera_rotation_representation,
            cam_translation_representation="extrinsics",
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

    def prepare_models(self):
        logger.info("Build models ...")

        # Ensures voxel embedder config matches the pixel denoiser model config
        if self.cfg.voxel_decoder_checkpoint is not None:
            # Ensure raster conditioning shape matches embedding dimension
            assert self.cfg.voxel_embedding_dim == self.cfg.denoiser.raster_cond_shape[0]

            # Update aggregation config feature dimension
            if self.cfg.denoiser.aggregation_config is not None:
               assert self.cfg.denoiser.aggregation_config.feature_dim == self.cfg.voxel_embedding_dim

            self.voxel_decoder = self.load_voxel_decoder()
            logger.info(f"Loaded voxel decoder from {self.cfg.voxel_decoder_checkpoint} ...")
        else:
            self.voxel_decoder = None 


        # Initialize models.
        self.models = nn.ModuleDict(
            {
                "denoiser": getattr(dit_pixel, self.cfg.denoiser.name)(**vars(self.cfg.denoiser)),

                "voxel_class_embedder": nn.Embedding(self.voxel_vocab_size, self.cfg.voxel_embedding_dim)
                if self.cfg.voxel_decoder_checkpoint is not None else None,
            }
        )
        if self.cfg.voxel_decoder_checkpoint is None:
            self.models.pop("voxel_class_embedder") #hacky
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
        x_0 = batch["pixel"]

        # build conditioning
        cond = {
            "action": batch["action"].clone(),
            "camera": batch["camera"].clone(),
        }
        # noise augmentation on voxel latents
        noise_vox = torch.randn_like(batch["voxel"].float())
        noise_vox = torch.clamp(noise_vox, -self.flow_euler.cfg.noise_abs_max, self.flow_euler.cfg.noise_abs_max)
        cond["voxel_latents"] = (1 - self.flow_euler.cfg.noise_cond) * batch["voxel"]  + self.flow_euler.cfg.noise_cond * noise_vox
        # Handle voxel latents or decoded voxels
        if self.cfg.voxel_decoder_checkpoint is not None:
            cond["voxel_latents"]= self.decode_voxel_latents_batch(cond["voxel_latents"], self.voxel_decoder, batch_size=self.cfg.voxel_decoder_batch_size)

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
        if self.cfg.loss_type == "mse":
            terms.loss = F.mse_loss(pred, target)
            if verbose:
                with torch.no_grad():
                    loss_per_timestep = F.mse_loss(
                        pred, target, reduction="none").mean(dim=list(range(2, len(pred.shape))))
        elif self.cfg.loss_type == "ramp_mse":
            loss_weights = (0.9 * (1 - t) + 0.1).detach()
            loss_per_timestep = F.mse_loss(pred, target, reduction="none").mean(dim=list(range(2, len(pred.shape))))
            terms.loss = (loss_per_timestep * loss_weights).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.cfg.loss_type}")

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

        return terms, {}

    def load_voxel_decoder(self):
        """Load frozen voxel VAE decoder from checkpoint."""
        import json

        # Load voxel class dictionary
        classdict_path = os.path.join(self.dataset.root, 'mt_voxel_classdict.json')
        with open(classdict_path, 'r') as f:
            voxel_classdict = json.load(f)
        vocab_size = len(voxel_classdict['node_classes'])
        if vocab_size != self.cfg.voxel_decoder.out_channels:
            raise ValueError(f"Voxel decoder output channels {self.cfg.voxel_decoder.out_channels} does not match voxel "
                             f"class vocab size {vocab_size} of the training dataset.")
        self.voxel_vocab_size = vocab_size  # Store for embedding layer initialization

        # Create decoder model
        decoder = vae_voxel.ResNet3dDecoder(
            out_channels=self.cfg.voxel_decoder.out_channels,
            latent_channels=self.cfg.voxel_decoder.latent_channels,
            channels=self.cfg.voxel_decoder.channels,
            num_res_blocks=self.cfg.voxel_decoder.num_res_blocks,
            num_res_blocks_middle=self.cfg.voxel_decoder.num_res_blocks_middle,
        )

        # Load weights and freeze
        checkpoint = safetensors.torch.load_file(self.cfg.voxel_decoder_checkpoint)
        if list(checkpoint.keys())[0].startswith("_orig_mod."):
            checkpoint = {k.split("_orig_mod.")[1] : v for k, v in checkpoint.items()}
        decoder.load_state_dict(checkpoint)
        decoder.eval()
        for param in decoder.parameters():
            param.requires_grad = False

        return decoder.to(self.accelerator.device)

    def decode_voxel_latents_batch(self, voxel_latents, decoder, batch_size=None):
        """Decode normalized voxel latents to (B,T,C,64,64,64) voxel tensors."""

        # Unnormalize voxel latents
        if self.dataset.normalization['voxel_latent']:
            norm_factors = self.dataset.normalization['voxel_latent']
            voxel_mean = norm_factors['mean'].view(1, 1, -1, 1, 1, 1).to(voxel_latents.device)
            voxel_std = norm_factors['std'].view(1, 1, -1, 1, 1, 1).to(voxel_latents.device)
            voxel_latents = voxel_latents * voxel_std + voxel_mean

        # Decode latents to voxel classes (vectorized)
        B, T = voxel_latents.shape[:2]
        voxel_latents_flat = rearrange(voxel_latents, "B T C X Y Z -> (B T) C X Y Z")

        batch_size = len(voxel_latents_flat) if batch_size is None else batch_size
        voxel_classes = []
        for i in range(0, len(voxel_latents_flat), batch_size):
            with torch.no_grad(), self.accelerator.autocast():
                z = voxel_latents_flat[i:i+batch_size] #.to(self.weight_dtype)
                logits = decoder(z)  # (bs, vocab_size, 64, 64, 64)
                voxel_classes.append(torch.argmax(logits, dim=-1))  # (bs, 64, 64, 64)

        # Re-encode with learned embeddings
        # Apply embedding to class indices: (B*T, 64, 64, 64) -> (B*T, 64, 64, 64, embed_dim)
        voxel_classes = torch.cat(voxel_classes, dim=0) # (B*T, 64, 64, 64)
        embedded_voxels = self.models["voxel_class_embedder"](voxel_classes)  # (B*T, 64, 64, 64, embed_dim)

        # Rearrange to match expected format: (B, T, embed_dim, 64, 64, 64)
        embedded_voxels = rearrange(embedded_voxels, "(B T) X Y Z C -> B T C X Y Z", B=B, T=T)

        return embedded_voxels


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
    trainer = PixelDiffuserTrainer(args, accelerator)
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
