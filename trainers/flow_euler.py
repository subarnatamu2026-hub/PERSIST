import dataclasses
from typing import Annotated, Literal, Tuple, Union

import numpy as np
import torch
import tyro
from easydict import EasyDict as edict
from einops import rearrange, repeat
from tqdm import tqdm

from trainers import BaseTrainingArgs


# fmt:off
@dataclasses.dataclass
class DiffusionTeacherForcingArgs:
    """Configuration for Teacher Forcing."""

    @property
    def mode(self) -> str:
        return "teacher-forcing"

@dataclasses.dataclass
class DiffusionForcingArgs:
    """Configuration for Diffusion Forcing."""

    sigma_min: float = 1e-5
    noise_cond: float = 0.015
    noise_abs_max: float = 20.0
    t_schedule: Literal["uniform", "logitNormal"] = "logitNormal"
    t_schedule_mean: float = 0.0
    """Only used if t_schedule is logitNormal"""
    t_schedule_std: float = 1.0
    """Only used if t_schedule is logitNormal"""
    rescale_t_inference: float = 1.0
    min_noise_level_inference: float = 0.015

    @property
    def mode(self) -> str:
        return "diffusion-forcing"

@dataclasses.dataclass(kw_only=True)
class FlowEulerTrainingArgs(BaseTrainingArgs):
    diffusion: Union[
        Annotated[DiffusionTeacherForcingArgs, tyro.conf.subcommand("teacher-forcing")],
        Annotated[DiffusionForcingArgs, tyro.conf.subcommand("diffusion-forcing")],
    ]
    """Mode of diffusion to use."""

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

    normalize_latents: bool = True

    log_train_metrics: bool = True
    """Whether to log extra training metrics (binned losses by diffusion time etc)."""

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
        # if self.i_val:
        #     assert self.i_log % self.i_val == 0, "i_log must be a multiple of i_val"

class FlowEuler:

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [seq_length x batch_size x ...] tensor of noiseless inputs.
            t: The [seq_length, batch_size] tensor of diffusion steps [0-1].
            noise: The [seq_length x batch_size x ...] tensor of noise to add.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(*t.shape, *[1 for _ in range(len(x_0.shape) - 2)])
        x_t = (1 - t) * x_0 + (self.cfg.sigma_min + (1 - self.cfg.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(*t.shape, *[1 for _ in range(len(x_t.shape) - 2)])
        x_0 = (x_t - (self.cfg.sigma_min + (1 - self.cfg.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v_from_noise(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.cfg.sigma_min) * noise - x_0

    def get_v_from_xt(self, x_t, x_0, t):
        """
        Compute the velocity of the diffusion process at time t.
        """
        t = t.view(*t.shape, *[1 for _ in range(len(x_t.shape) - 2)])
        v = ((1 - self.cfg.sigma_min) * x_t - x_0) / (self.cfg.sigma_min + (1 - self.cfg.sigma_min) * t)
        return v

    def sample_t(self, batch_size: int, seq_length:int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.cfg.t_schedule == 'uniform':
            if self.cfg.mode == 'teacher-forcing':
                t = torch.rand(batch_size).to(self.device)
            elif self.cfg.mode == 'diffusion-forcing':
                t = torch.rand(batch_size, seq_length).to(self.device)
        elif self.cfg.t_schedule == 'logitNormal':
            mean = self.cfg.t_schedule_mean
            std = self.cfg.t_schedule_std
            if self.cfg.mode == 'teacher-forcing':
                t = torch.sigmoid(torch.randn(batch_size) * std + mean).to(self.device)
            elif self.cfg.mode == 'diffusion-forcing':
                t = torch.sigmoid(torch.randn(batch_size, seq_length) * std + mean).to(self.device)
        else:
            raise ValueError(f"Unknown t_schedule: {self.cfg.t_schedule}")
        return t

    def sample_xprev(self, x_t, v, t, t_prev):
        dt = (t - t_prev).view(*t.shape, *([1] * (x_t.ndim - 2)))
        pred_x_prev = x_t - dt * v
        return pred_x_prev

    def sample_x0(self, x_t, t, v):
        t = t.view(*t.shape, *([1] * (x_t.ndim - 2)))
        x_0 = (1 - self.cfg.sigma_min) * x_t - (self.cfg.sigma_min + (1 - self.cfg.sigma_min) * t) * v
        return x_0

    def schedule_timesteps(self, num_inference_steps):
        steps = np.linspace(1, 0, num_inference_steps)
        steps = self.cfg.rescale_t_inference * steps / (1 + (self.cfg.rescale_t_inference - 1) * steps)
        return steps

    def denoise_parallel(self, model, noise, cond, steps, denoiser_pred='v', store_intermediate_steps=False, verbose=True):
        x_t = noise
        t_seq = self.schedule_timesteps(num_inference_steps=steps + 1)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"sample": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Parallel diffusion", disable=not verbose):
            if denoiser_pred == 'v':
                pred_v = model(x_t, t, cond)
                x_t = self.sample_xprev(x_t, pred_v, t, t_prev)
            elif denoiser_pred == 'x0':
                pred_x_0 = model(x_t, t, cond)
                pred_v = self.get_v_from_xt(x_t, pred_x_0, t)
                x_t = self.sample_xprev(x_t, pred_v, t, t_prev)
            else:
                raise ValueError(f"Unknown denoiser_pred: {denoiser_pred}")

            if store_intermediate_steps:
                ret.pred_x_t.append(x_t.cpu())
                if denoiser_pred == 'v':
                    pred_x_0 = self.sample_x0(x_t, t, pred_v)
                ret.pred_x_0.append(pred_x_0.cpu())
        ret.sample = x_t.cpu()
        if store_intermediate_steps:
            ret.pred_x_0 = rearrange(torch.cat(ret["pred_x_0"], dim=1), "B (T tau) ... -> B T tau ...", T=x_t.shape[1], tau=len(steps))
            ret.pred_x_t = rearrange(torch.cat(ret["pred_x_t"], dim=1), "B (T tau) ... -> B T tau ...", T=x_t.shape[1], tau=len(steps))
        return ret

    def denoise_autoreg(self, model, ep_length, seq_length, diffusion_steps, stable_t, cond, x_start, repeat_cond=False,
                        denoiser_pred='v', reencode_fn=None, store_intermediate_steps=False, verbose=True):
        if isinstance(x_start, torch.Tensor):
            x_tuple = (x_start,)
            x_shapes = (x_start.shape[2:],)
        elif isinstance(x_start, tuple):
            x_tuple = x_start
            x_shapes = tuple(x.shape[2:] for x in x_start)
        else:
            raise ValueError("x_start must be a tensor or a tuple of tensors")

        if store_intermediate_steps:
            assert isinstance(x_start, torch.Tensor), "Storing intermediate steps is only supported for single-tensor x_start currently."

        B, Tprompt = x_tuple[0].shape[0], x_tuple[0].shape[1]
        device = x_tuple[0].device

        if Tprompt == 0:
            from_noise = True
        else:
            from_noise = False
            assert Tprompt < seq_length
        for cond_type in cond.keys():
            cond[cond_type] = cond[cond_type][:, :ep_length]
            if repeat_cond:
                assert cond[cond_type].shape[0] == 1, "When repeat_cond is True, cond should have batch size 1"

        t_seq = self.schedule_timesteps(num_inference_steps=diffusion_steps + 1)
        ret = edict({"sample": None, "pred_x_t": []*len(x_tuple), "pred_x_0": []*len(x_tuple)})
        for i in tqdm(range(Tprompt, ep_length), desc="Autoreg diffusion", disable=not verbose):
            start_seq_i = max(0, i + 1 - seq_length)
            noise_tuple = tuple(torch.randn(B, 1, *x_shape, device=device) for x_shape in x_shapes)
            noise_tuple = tuple(torch.clamp(noise, -self.cfg.noise_abs_max, self.cfg.noise_abs_max) for noise in noise_tuple)
            x_tuple = tuple(torch.cat([x, noise], dim=1) for x, noise in zip(x_tuple, noise_tuple))

            cond_i = {k: v[:, start_seq_i : i+1] for k, v in cond.items()}
            if repeat_cond:
                cond_i = {k: repeat(v, "1 ... -> B ...", B=B) for k, v in cond_i.items()}
            t_cst = torch.full((B, i), stable_t, device=device)
            for noise_idx in range(0, diffusion_steps):
                t = torch.full((B, 1), t_seq[noise_idx], device=device)
                t_prev = torch.full((B, 1), t_seq[noise_idx + 1], device=device)
                t = torch.cat([t_cst, t], dim=1)
                t_prev = torch.cat([t_cst, t_prev], dim=1)

                #sliding window
                x_curr = tuple(x.clone() for x in x_tuple)
                x_curr = tuple(x_cur[:, start_seq_i:] for x_cur in x_curr)
                t = t[:, start_seq_i:]
                t_prev = t_prev[:, start_seq_i:]

                if from_noise and i == 0:
                    pass
                else:
                    added_noise_tuple = tuple(torch.randn_like(x_cur[:, :-1] ) for x_cur in x_curr)
                    added_noise_tuple = tuple(torch.clamp(added_noise, -self.cfg.noise_abs_max, self.cfg.noise_abs_max) for added_noise in added_noise_tuple)
                    for x_cur, added_noise in zip(x_curr, added_noise_tuple):
                        x_cur[:, :-1] = self.diffuse(x_cur[:, :-1], t[:, :-1], added_noise)

                x_input = x_curr[0] if len(x_curr) == 1 else x_curr
                if denoiser_pred == 'v':
                    pred_v = model(x_input, t, cond_i)
                    pred_v = (pred_v,) if isinstance(pred_v, torch.Tensor) else pred_v
                elif denoiser_pred == 'x0':
                    pred_x_0 = model(x_input, t, cond_i)
                    pred_x_0 = (pred_x_0,) if isinstance(pred_x_0, torch.Tensor) else pred_x_0
                    pred_v = tuple(self.get_v_from_xt(x_cur, px_0, t) for x_cur, px_0 in zip(x_curr, pred_x_0))
                else:
                    raise ValueError(f"Unknown denoiser_pred: {denoiser_pred}")
                for pv, x, x_cur in zip(pred_v, x_tuple, x_curr):
                    x[:, -1:] = self.sample_xprev(x_cur, pv, t, t_prev)[:, -1:]

                if store_intermediate_steps:
                    if denoiser_pred == 'v':
                        pred_x_0 = tuple(self.sample_x0(x_cur[:, -1:], t[:, -1:], pv[:, -1:]) for pv, x_cur in zip(pred_v, x_curr))
                    for (px_0, x) in zip(pred_x_0, x_tuple):
                        ret["pred_x_0"].append(px_0.cpu())
                        ret["pred_x_t"].append(x[:, -1:].cpu())
            if reencode_fn is not None:
                for x in x_tuple:
                    x[:, -1:] = reencode_fn(x[:, -1:])

        ret.sample = tuple(x.cpu() for x in x_tuple)
        if len(x_tuple) == 1:
            ret.sample = ret.sample[0]
        if store_intermediate_steps:
            ret.pred_x_0 = rearrange(torch.cat(ret["pred_x_0"], dim=1), "B (T tau) ... -> B T tau ...", T=ep_length - Tprompt, tau=diffusion_steps)
            ret.pred_x_t = rearrange(torch.cat(ret["pred_x_t"], dim=1), "B (T tau) ... -> B T tau ...", T=ep_length - Tprompt, tau=diffusion_steps)
        return ret
