# fmt:off
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
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader

from data_loaders.minetest_voxel_dataset import MinetestVoxel
from models import vae_voxel
from trainers import BaseTrainingArgs, BaseTrainer
from utils.loss_util import generalized_dice_loss, proximity_mask

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass(kw_only=True)
class TrainingArgs(BaseTrainingArgs):
    """Configuration for training the model."""

    encoder: vae_voxel.ResNetEncoderArgs
    """Encoder configuration"""

    decoder: vae_voxel.ResNetDecoderArgs
    """Decoder configuration"""

    run_name: str = "debug"
    """Experiment name"""

    random_seed: int = 42

    output_dir: str = "outputs/runs/vae_voxel"
    """Output directory"""

    remote_output_dir: str | None = None
    """For Azure only. Remote directory for saving the output."""

    dataset_path: str = "datasets"
    """Path to training dataset"""

    min_level_quality_score: float = 1.0
    """Levels below this quality score will be filtered out from the dataset"""

    batch_size: int = 1
    """Batch size for training (per device)"""

    grad_accumulation_steps: int = 1

    total_steps: int | None = None
    """Total training steps"""

    total_grad_steps: int | None = None

    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"

    # Loss related

    loss_type: Literal["ce", "dice", "proximity_reweighted_ce"] = "ce"

    loss_temperature: float = 2.0
    """Temperature for proximity reweighted loss"""

    lambda_kl: float = 1e-06

    max_grad_norm: float = 1.0

    learning_rate: float = 1e-04

    # Frequency

    i_print: int = 100

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


class VoxelVaeTrainer(BaseTrainer):

    def __init__(self, cfg: TrainingArgs, accelerator: Accelerator):
        super().__init__(cfg, accelerator)
        if self.cfg.loss_type == "proximity_reweighted_ce":
            self.proximity_weights = proximity_mask((cfg.encoder.input_x, cfg.encoder.input_y, cfg.encoder.input_z),
                                                    temperature=cfg.loss_temperature).to(self.accelerator.device)

    def prepare_dataset(self):
        logger.info("Build Datasets ...")
        self.dataset = MinetestVoxel(
            self.cfg.dataset_path,
            min_level_quality_score=self.cfg.min_level_quality_score,
            resolution=self.cfg.encoder.input_x,
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

        self.val_dataset = MinetestVoxel(
            self.cfg.dataset_path,
            min_level_quality_score=self.cfg.min_level_quality_score,
            resolution=self.cfg.encoder.input_x,
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
        vocab_size = self.dataset.vocab_size["node_classes"]
        if vocab_size != self.cfg.encoder.in_channels:
            raise ValueError(f"Voxel encoder input channels {self.cfg.encoder.in_channels} does not match voxel "
                             f"class vocab size {vocab_size} of the training dataset.")
        if vocab_size != self.cfg.decoder.out_channels:
            raise ValueError(f"Voxel decoder output channels {self.cfg.decoder.out_channels} does not match voxel "
                             f"class vocab size {vocab_size} of the training dataset.")
        self.models = nn.ModuleDict(
            {
                "encoder": getattr(vae_voxel, self.cfg.encoder.name)(
                    **vars(self.cfg.encoder)
                ),
                "decoder": getattr(vae_voxel, self.cfg.decoder.name)(
                    **vars(self.cfg.decoder)
                ),
            }
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
            targets: The [B x X x Y x Z] tensor of voxel classes.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        targets = batch["voxel_classes"].long()
        z, mean, logvar = self.models["encoder"](targets, sample_posterior=True, return_raw=True)
        logits = self.models["decoder"](z)

        terms = edict(loss=0.0)
        if self.cfg.loss_type == "ce":
            logits = rearrange(logits, "b x y z c -> (b x y z) c")
            targets = rearrange(targets, "b x y z-> (b x y z)")
            terms["ce"] = F.cross_entropy(logits, targets, reduction="mean")
            terms["loss"] = terms["loss"] + terms["ce"]
        elif self.cfg.loss_type == "proximity_reweighted_ce":
            X, Y, Z = targets.shape[1:]
            logits = rearrange(logits, "b x y z c -> (b x y z) c")
            targets = rearrange(targets, "b x y z-> (b x y z)")
            loss_weights = self.proximity_weights.unsqueeze(0)
            loss = F.cross_entropy(logits, targets, reduction="none")
            terms["ce"] = (rearrange(loss, "(b x y z) -> b x y z", x=X, y=Y, z=Z)  * loss_weights).mean()
            terms["loss"] = terms["loss"] + terms["ce"]
        elif self.cfg.loss_type == "dice":
            terms["dice"] = generalized_dice_loss(logits, targets, epsilon=1e-6)
            terms["loss"] = terms["loss"] + terms["dice"]
        else:
            raise ValueError(f"Invalid loss type {self.cfg.loss_type}")
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
    trainer = VoxelVaeTrainer(args, accelerator)
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
