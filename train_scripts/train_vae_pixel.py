import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from easydict import EasyDict as edict
from loguru import logger
from torch.utils.data import DataLoader

from data_loaders.minetest_pixel_dataset import MinetestPixel
from models import vae_pixel
from trainers import BaseTrainingArgs, BaseTrainer

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass(kw_only=True)
class TrainingArgs(BaseTrainingArgs):
    """Configuration for training the pixel VAE model."""

    model: vae_pixel.ViTVaeArgs

    run_name: str = "debug"
    """Experiment name"""

    random_seed: int = 42

    output_dir: str = "outputs/runs/vae_pixel"
    """Output directory"""

    remote_output_dir: str | None = None
    """For Azure only. Remote directory for saving the output."""

    ckpt_to_load: str | None = None
    """Path to a specific accelerator checkpoint folder which we want to resume from"""

    dataset_path: str = "datasets"
    """Path to training dataset"""

    min_level_quality_score: float = 1.0
    """Levels below this quality score will be filtered out from the dataset"""

    batch_size: int = 1
    """Batch size for training (per device)"""

    grad_accumulation_steps: int = 1

    total_steps: int = 1_000_000
    """Total training steps"""

    total_grad_steps: int | None = None

    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"

    # Loss related

    # Using MSE loss for RGB reconstruction

    lambda_kl: float = 1e-06

    max_grad_norm: float = 1.0

    learning_rate: float = 1e-04

    # Frequency

    i_print: int = 10

    i_log: int = 100

    i_val : int = 50

    i_save: int = 10_000

    def __post_init__(self):
        super().__post_init__()
        self.log_train_metrics = False
        self.lr_warmup_steps = 0

        assert self.total_steps or self.total_grad_steps, "Either total_steps or total_grad_steps must be set."
        assert not (self.total_steps and self.total_grad_steps), "Only one of total_steps or total_grad_steps can be set."
        if self.total_grad_steps:
            self.total_steps = self.total_grad_steps * self.grad_accumulation_steps
        else:
            self.total_grad_steps = self.total_steps // self.grad_accumulation_steps
        if self.i_val:
            assert self.i_log % self.i_val == 0, "i_log must be a multiple of i_val"


class PixelVaeTrainer(BaseTrainer):

    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)

    def prepare_dataset(self):
        logger.info("Build Datasets ...")
        self.dataset = MinetestPixel(
            self.cfg.dataset_path,
            min_level_quality_score=self.cfg.min_level_quality_score,
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

        self.val_dataset = MinetestPixel(
            self.cfg.dataset_path,
            min_level_quality_score=self.cfg.min_level_quality_score,
            split="val",
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
        self.models = nn.ModuleDict(
            {"vae": getattr(vae_pixel, self.cfg.model.name)(**vars(self.cfg.model))}
        )

    def prepare_optimizer(self):
        logger.info("Initialize optimizer and lr scheduler ...")
        num_trainable_parameters = sum(p.numel() for p in self.models.parameters())
        logger.info(f"# of trainable parameters: {num_trainable_parameters / 1e6: .4f}M")
        self.optimizer = torch.optim.AdamW(
            self.models.parameters(), lr=self.cfg.learning_rate, weight_decay=0.0
        )

    def training_losses(self, batch, **kwargs) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            targets: The [B x C x H x W] tensor of RGB images in range [-1, 1].

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        targets = batch["image"]
        # Forward pass through VAE: returns (rec, post, latent)
        rec, post, latent = self.models["vae"](
            targets, targets
        )  # labels argument not used but required

        terms = edict(loss=0.0)

        # RGB reconstruction loss (MSE)
        terms["mse"] = F.mse_loss(rec, targets, reduction="mean")
        terms["loss"] = terms["loss"] + terms["mse"]

        # KL divergence loss - get mean and logvar from posterior distribution
        mean = post.mean
        logvar = post.logvar
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.cfg.lambda_kl * terms["kl"]

        return terms, {}


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
        # step_scheduler_with_optimizer=False # uncomment this if you want to use an lr scheduler
    )
    trainer = PixelVaeTrainer(args, accelerator)
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
