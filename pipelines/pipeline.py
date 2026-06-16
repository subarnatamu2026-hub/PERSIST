import time
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import *

import torch
from diffusers import DiffusionPipeline
from einops import rearrange
from loguru import logger
from torch import nn
from tqdm import tqdm

import models
from trainers.flow_euler import FlowEuler
from utils.camera_util import local_to_worldcam, rotation_6d_to_matrix

REQUIRED_MODEL_KEYS = [
    "voxel_encoder",
    "voxel_decoder",
    "pixel_vae",
    "camera_model",
    "voxel_denoiser",
    "pixel_denoiser",
    "voxel_class_embedder",
]

# The persist_S pipeline ships a single config (the "S" variant). The "XL" variant
# is the same config with the voxel denoiser patch_size lowered to 1 and the sampler
# noise conditioning disabled. Callers translate the chosen variant into the kwargs
# accepted by `BasePipeline.from_pretrained` via this helper.
PIPELINE_VARIANTS = ("S", "XL")


def pipeline_variant_overrides(variant: str):
    """Return ``(model_config_overrides, sampler_args)`` for a persist_S variant.

    - ``"S"``  -> no overrides (the values shipped in pipeline.json / model_configs.py).
    - ``"XL"`` -> voxel-denoiser ``patch_size`` 1 and sampler ``noise_cond`` 0.0.

    ``sampler_args`` is applied to both the voxel and pixel samplers.
    """
    variant = variant.upper()
    if variant == "S":
        return {}, {}
    if variant == "XL":
        return {"voxel_denoiser": {"patch_size": 1}}, {"noise_cond": 0.0}
    raise ValueError(f"Unknown pipeline variant '{variant}'. Expected one of {PIPELINE_VARIANTS}.")

@dataclass
class SamplerArgs:
    denoiser_pred: str = "v"
    sigma_min: float = 1e-5
    noise_cond: float = 0.0
    noise_abs_max: float = 20.0
    rescale_t_inference: float = 1.0
    min_noise_level_inference: float = 0.1
    steps: int = 20
    steps_t0: int = 20

class BasePipeline(DiffusionPipeline):

    def __init__(
            self,
            models: Union[Dict, nn.ModuleDict],
            accelerator=None,
            voxel_sampler=None,
            pixel_sampler=None,
            latent_normalization_stats=None,
            batch_size_voxel_vae: int = 8,
            batch_size_pixel_vae: int = 16,
            reencode_voxels: bool = True,
            wait_every: int = 200, # if accelerator is provided, how often to sync GPUs during autoregressive rollout (in number of frames)
    ):
        missing = [k for k in REQUIRED_MODEL_KEYS if k not in models]
        if missing:
            raise ValueError(
                f"`models` dict is missing required keys: {missing}. "
                f"Expected keys: {REQUIRED_MODEL_KEYS}"
            )
        super().__init__()
        self.models = models
        self.accelerator = accelerator
        self.register_modules(**models)
        self.voxel_sampler = voxel_sampler
        self.pixel_sampler = pixel_sampler
        self.latent_norm = latent_normalization_stats
        self.bs_voxel_vae = batch_size_voxel_vae
        self.bs_pixel_vae = batch_size_pixel_vae
        self.reencode_voxels = reencode_voxels
        self.wait_every = wait_every

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Self":
        """
        Load a pretrained model.
        """
        import os
        import json
        import importlib.util
        import sys
        from safetensors.torch import load_file
        is_local = os.path.exists(f"{path}/pipeline.json")
        if not is_local:
            # Treat `path` as a Hugging Face Hub repo id and download the whole
            # pipeline folder (pipeline.json, model_configs.py, ckpts/*.safetensors).
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id=path, revision=kwargs.pop("revision", None))
        config_file = f"{path}/pipeline.json"

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        device = kwargs.pop("device", torch.device("cpu"))

        # Checkpoints are always supplied by the caller (one path per model key);
        # this pipeline folder never ships a `ckpts/` directory.
        custom_checkpoint_cfg = kwargs.pop("custom_checkpoint_cfg", None) or {}
        # Optional per-model architecture overrides, e.g. {"voxel_denoiser": {"patch_size": 1}}.
        model_config_overrides = kwargs.pop("model_config_overrides", None) or {}
        voxel_sampler_args = kwargs.pop("voxel_sampler_args", None)
        pixel_sampler_args = kwargs.pop("pixel_sampler_args", None)
        if voxel_sampler_args is not None:
            args['voxel_sampler']['args'].update(voxel_sampler_args)
        if pixel_sampler_args is not None:
            args['pixel_sampler']['args'].update(pixel_sampler_args)

        # Dynamically import model_configs.py from the checkpoint folder
        model_cfg_path = os.path.join(path, "model_configs.py")
        spec = importlib.util.spec_from_file_location("model_configs", model_cfg_path)
        model_configs = importlib.util.module_from_spec(spec)
        sys.modules["model_configs"] = model_configs
        spec.loader.exec_module(model_configs)

        # Instantiate models into pipeline
        _models = {}
        for k, v in args['models'].items():
            config = getattr(model_configs, f"{v}Args")()
            for field_name, field_value in model_config_overrides.get(k, {}).items():
                setattr(config, field_name, field_value)
            _models[k] = getattr(models, config.name)(**asdict(config))
            checkpoint_path = custom_checkpoint_cfg.get(k)
            if checkpoint_path is None:
                logger.warning("{}: no checkpoint path provided, using randomly-initialised weights.".format(k))
            elif os.path.exists(checkpoint_path):
                checkpoint = load_file(checkpoint_path)
                if list(checkpoint.keys())[0].startswith("_orig_mod."):
                    checkpoint = {k.split("_orig_mod.")[1]: v for k, v in checkpoint.items()}
                _models[k].load_state_dict(checkpoint, strict=False)
            else:
                logger.warning("{}: No checkpoint found at {}".format(k, checkpoint_path))

        new_pipeline = cls(_models, accelerator=kwargs.get("accelerator", None))

        # Instantiate samplers
        voxel_sampler_config = SamplerArgs(**args['voxel_sampler']['args'])
        new_pipeline.voxel_sampler = FlowEuler(voxel_sampler_config, device)
        pixel_sampler_config = SamplerArgs(**args['pixel_sampler']['args'])
        new_pipeline.pixel_sampler = FlowEuler(pixel_sampler_config, device)
        new_pipeline.latent_norm = {
            "voxel": {
                "mean": torch.tensor(args["latent_normalization_stats"]["voxel"]["mean"]),
                "std": torch.tensor(args["latent_normalization_stats"]["voxel"]["std"]),
            },
            "pixel": {
                "mean": torch.tensor(args["latent_normalization_stats"]["pixel"]["mean"]),
                "std": torch.tensor(args["latent_normalization_stats"]["pixel"]["std"]),
            }
        }

        # Move to device
        new_pipeline.to(device)  # 6GB footprint
        # notes:
        # new_pipeline.reset_device_map(), print(new_pipeline.hf_device_map) # -> is broken, always return none

        # Add other arguments
        new_pipeline.bs_voxel_vae = args["batch_size_voxel_vae"]
        new_pipeline.bs_pixel_vae = args["batch_size_pixel_vae"]
        new_pipeline.reencode_voxels = args["reencode_voxels"]

        return new_pipeline

    def encode_voxels(self, voxel):
        voxel = voxel.to(self.device, dtype=torch.long)
        B, T = voxel.shape[:2]
        voxel = rearrange(voxel, "B T X Y Z -> (B T) X Y Z")
        z = []
        for i in range(0, len(voxel), self.bs_voxel_vae):
            with torch.no_grad():
                z.append(self.voxel_encoder(voxel[i:i+self.bs_voxel_vae]))

        z = torch.cat(z, dim=0)
        z = rearrange(z, "(B T) C X Y Z -> B T C X Y Z", B=B, T=T)

        # normalize
        norm_factors = self.latent_norm['voxel']
        voxel_mean = norm_factors['mean'].view(1, 1, -1, 1, 1, 1)
        voxel_std = norm_factors['std'].view(1, 1, -1, 1, 1, 1)
        z = (z - voxel_mean) / voxel_std
        return z

    def decode_voxels(self, voxel_latents):
        # Unnormalize voxel latents
        voxel_latents = voxel_latents.to(self.device)
        norm_factors = self.latent_norm['voxel']
        voxel_mean = norm_factors['mean'].view(1, 1, -1, 1, 1, 1)
        voxel_std = norm_factors['std'].view(1, 1, -1, 1, 1, 1)
        voxel_latents = voxel_latents * voxel_std + voxel_mean

        # Decode latents to voxel classes (vectorized)
        B, T = voxel_latents.shape[:2]
        voxel_latents = rearrange(voxel_latents, "B T C X Y Z -> (B T) C X Y Z")

        x = []
        for i in range(0, len(voxel_latents), self.bs_voxel_vae):
            with torch.no_grad():
                z = voxel_latents[i:i+self.bs_voxel_vae]
                logits = self.voxel_decoder(z)  # (bs, vocab_size, 64, 64, 64)
                x.append(torch.argmax(logits, dim=-1).to(torch.int16))  # (bs, 64, 64, 64)

        x = torch.cat(x, dim=0) # (B*T, X Y Z)
        x = rearrange(x, "(B T) X Y Z -> B T X Y Z", B=B, T=T)
        return x

    def encode_pixels(self, pixel):
        pixel = pixel.to(self.device)
        B, T = pixel.shape[:2]
        pixel = rearrange(pixel, "B T C H W-> (B T) C H W")

        z = []
        for i in range(0, len(pixel), self.bs_pixel_vae):
            with torch.no_grad():
                posterior = self.pixel_vae.encode(pixel[i:i+self.bs_pixel_vae])
                z.append(posterior.mode())

        z = torch.cat(z, dim=0)
        z = rearrange(z, "(B T) (H W) C -> B T C H W", B=B, T=T, H=self.pixel_vae.seq_h ,W=self.pixel_vae.seq_w)

        # normalize
        norm_factors = self.latent_norm['pixel']
        pixel_mean = norm_factors['mean'].view(1, 1, -1, 1, 1)
        pixel_std = norm_factors['std'].view(1, 1, -1, 1, 1)
        z = (z - pixel_mean) / pixel_std
        return z

    def decode_pixels(self, pixel_latents):
        # Unnormalize pixel latents
        pixel_latents = pixel_latents.to(self.device)
        norm_factors = self.latent_norm['pixel']
        pixel_mean = norm_factors['mean'].view(1, 1, -1, 1, 1)
        pixel_std = norm_factors['std'].view(1, 1, -1, 1, 1)
        pixel_latents = pixel_latents * pixel_std + pixel_mean

        # Decode latents to pixels (vectorized)
        B, T = pixel_latents.shape[:2]
        pixel_latents = rearrange(pixel_latents, "B T C H W -> (B T) (H W) C")

        x = []
        for i in range(0, len(pixel_latents), self.bs_pixel_vae):
            with torch.no_grad():
                z = pixel_latents[i:i+self.bs_pixel_vae]
                xx = self.pixel_vae.decode(z)
                x.append(xx)

        x = torch.cat(x, dim=0) # (B*T, C, H, W)
        x = rearrange(x, "(B T) C H W -> B T C H W", B=B, T=T)
        return x

    def prepare_pix_denoiser_cond(self, rollout, global_ts, use_kv_cache=False):

        if use_kv_cache:
            # Single-frame conditioning — past context lives in the KV cache
            embedded_voxels = self.voxel_class_embedder(rollout["voxel"][:, -1:].to(self.device, dtype=torch.long))
            embedded_voxels = rearrange(embedded_voxels, "B T X Y Z C -> B T C X Y Z")
            camera = rollout["camera"][:, -1:].to(self.device)
            camera = self.project_voxel_local_to_worldcam(camera)
            cond = {
                "action": rollout["action"][:, -1:].to(self.device),
                "camera": camera,
                "voxel_latents": embedded_voxels,
            }
            return cond

        # Check sequence integrity
        assert rollout["action"].shape[1] == global_ts + 2 # (t_{-1}, t_{0}, ..., t_{global_ts})
        assert rollout["camera"].shape[1] == global_ts + 2 # (t_{-1}, t_{0}, ..., t_{global_ts})
        assert rollout["voxel"].shape[1] == global_ts + 2 #(t_{-1}, t_{0}, ..., t_{global_ts-1})
        if "pixel_latents" in rollout:
            assert rollout["pixel_latents"].shape[1] == global_ts + 1 # (t_{-1}, t_{0}, ..., t_{global_ts-1})

        B, C, H, W = (rollout["action"].shape[0], self.pixel_denoiser.in_channels, self.pixel_denoiser.input_h, self.pixel_denoiser.input_w)
        shift_vox = shift_act = shift_cam = shift_pix = 0

        cond_len_pix = self.pixel_denoiser.context_window_size - 1
        cond_len_other = self.pixel_denoiser.context_window_size
        if "pixel_latents" in rollout:
            x = rollout["pixel_latents"][:, shift_pix:][:, -cond_len_pix:]
        else:
            x = torch.empty((B, 0, C, H, W))

        embedded_voxels = self.voxel_class_embedder(rollout["voxel"][:, shift_vox:][:, -cond_len_other:].to(self.device, dtype=torch.long))
        embedded_voxels = rearrange(embedded_voxels, "B T X Y Z C -> B T C X Y Z")

        # apply camera transformation to match openCV convention
        camera = rollout["camera"][:, shift_cam:][:, -cond_len_other:].to(self.device)
        camera = self.project_voxel_local_to_worldcam(camera)

        cond = {
            "x" : x.to(self.device),
            "action": rollout["action"][:, shift_act:][:, -cond_len_other:].to(self.device),
            "camera": camera,
            "voxel_latents" : embedded_voxels,
        }
        return cond

    def prepare_vox_denoiser_cond(self, rollout, global_ts):
        raise NotImplementedError("Pipeline specific.")

    def prepare_camera_cond(self, rollout, global_ts):
        raise NotImplementedError("Pipeline specific.")

    @staticmethod
    def project_voxel_local_to_worldcam(camera):
        B, T = camera.shape[0], camera.shape[1]
        camera = rearrange(camera.clone(), "B T D -> (B T) D")
        R = rotation_6d_to_matrix(camera[..., :6])
        t = local_to_worldcam(camera[..., -4:-1], R)
        camera[..., -4:-1] = t
        camera = rearrange(camera, "(B T) D -> B T D", B=B, T=T)
        return camera

    @staticmethod
    def denoise_one_frame(model, sampler, cond, global_ts, enable_grad_at_step=None):
        steps = sampler.cfg.steps if global_ts > 0 else sampler.cfg.steps_t0
        with torch.no_grad():
            x = cond.pop("x")
            B, Tprompt, x_shape = x.shape[0], x.shape[1], x.shape[2:]

            # 1. add stable noise to existing frames
            t_cst = torch.full((B, Tprompt), sampler.cfg.min_noise_level_inference, device=x.device)
            if Tprompt > 0 and sampler.cfg.min_noise_level_inference > 0:
                noise_cst = torch.randn_like(x)
                noise_cst = torch.clamp(noise_cst, -sampler.cfg.noise_abs_max, sampler.cfg.noise_abs_max)
                x = sampler.diffuse(x, t_cst, noise_cst)

            # 2. initialise new frame as pure noise
            noise = torch.randn(B, 1, *x_shape, device=x.device)
            noise = torch.clamp(noise, -sampler.cfg.noise_abs_max, sampler.cfg.noise_abs_max)
            x = torch.cat([x, noise], dim=1)

            # 3. denoise frame
            grad_frame = None
            t_seq = sampler.schedule_timesteps(num_inference_steps=steps + 1)
            for noise_idx in range(0, steps):
                t = torch.full((B, 1), t_seq[noise_idx], device=x.device)
                t_prev = torch.full((B, 1), t_seq[noise_idx + 1], device=x.device)
                t = torch.cat([t_cst, t], dim=1)
                t_prev = torch.cat([t_cst, t_prev], dim=1)
                if sampler.cfg.denoiser_pred == 'v':
                    if noise_idx == enable_grad_at_step:
                        with torch.enable_grad():
                            grad_frame = model(x.clone(), t.clone(), cond, global_ts)
                            pred_v = grad_frame.clone().detach()
                            grad_frame = grad_frame[:, -1:]
                    else:
                        pred_v = model(x, t, cond, global_ts)
                elif sampler.cfg.denoiser_pred == 'x0':
                    if noise_idx == enable_grad_at_step:
                        with torch.enable_grad():
                            grad_frame = model(x.clone(), t.clone(), cond, global_ts)
                            pred_x_0 = grad_frame.clone().detach()
                            grad_frame = grad_frame[:, -1:]
                    else:
                        pred_x_0 = model(x, t, cond, global_ts)
                    pred_v = sampler.get_v_from_xt(x, pred_x_0, t)
                else:
                    raise ValueError(f"Unknown denoiser_pred: {sampler.cfg.denoiser_pred}")
                x[:, -1:] = sampler.sample_xprev(x, pred_v, t, t_prev)[:, -1:]

        return x[:, -1:], {"pred": grad_frame, "noise": noise}

    @staticmethod
    def denoise_one_frame_kv_cache(model, sampler, cond, global_ts, x_shape, enable_grad_at_step=None):
        """
        Differences from denoise_one_frame():
            - x does not need to be part of cond, it is created on the spot.
            - x_shape needs to be parsed in.
            - cond should only contain the conditions for the current frame, i.e. each item in cond has shape (B, 1, ...)
        """

        steps = sampler.cfg.steps if global_ts > 0 else sampler.cfg.steps_t0
        with torch.no_grad():
            # 1. initialise new frame as pure noise
            B = next(iter(cond.values())).shape[0]
            x = torch.randn(B, 1, *x_shape, device=sampler.device)
            x = torch.clamp(x, -sampler.cfg.noise_abs_max, sampler.cfg.noise_abs_max)

            # 2. denoise frame
            grad_frame = None
            noise = x.clone().detach() if enable_grad_at_step is not None else None
            t_seq = sampler.schedule_timesteps(num_inference_steps=steps + 1)
            for noise_idx in range(0, steps):
                t = torch.full((B, 1), t_seq[noise_idx], device=x.device)
                t_prev = torch.full((B, 1), t_seq[noise_idx + 1], device=x.device)
                if sampler.cfg.denoiser_pred == 'v':
                    if noise_idx == enable_grad_at_step:
                        with torch.enable_grad():
                            grad_frame = model(x.clone(), t.clone(), cond, global_ts)
                            pred_v = grad_frame.clone().detach()
                    else:
                        pred_v = model(x, t, cond, global_ts)
                elif sampler.cfg.denoiser_pred == 'x0':
                    if noise_idx == enable_grad_at_step:
                        with torch.enable_grad():
                            grad_frame = model(x.clone(), t.clone(), cond, global_ts)
                            pred_x_0 = grad_frame.clone().detach()
                    else:
                        pred_x_0 = model(x, t, cond, global_ts)
                    pred_v = sampler.get_v_from_xt(x, pred_x_0, t)
                else:
                    raise ValueError(f"Unknown denoiser_pred: {sampler.cfg.denoiser_pred}")
                x = sampler.sample_xprev(x, pred_v, t, t_prev)

        return x, {"pred": grad_frame, "noise": noise}

    @staticmethod
    def kv_cache_commit(model, sampler, x, cond, global_ts):
        """Commit a denoised (or re-encoded) frame to the KV cache.

        Runs one forward pass at the context noise level so the cache
        stores representations of the clean frame, not intermediate
        denoising states.
        """
        with torch.no_grad():
            B = x.shape[0]
            t_cst = torch.full((B, 1), sampler.cfg.min_noise_level_inference, device=x.device)
            if sampler.cfg.min_noise_level_inference > 0:
                noise_cst = torch.randn_like(x)
                noise_cst = torch.clamp(noise_cst, -sampler.cfg.noise_abs_max, sampler.cfg.noise_abs_max)
                x = sampler.diffuse(x.clone(), t_cst, noise_cst)
            cond_ = {k: v.clone() for k, v in cond.items()}
            _ = model(x, t_cst, cond_, global_ts)

    @staticmethod
    def prefill_kv_cache(model, sampler, cond, global_start_idx):
        """Reset and repopulate KV caches with a full-window forward pass.

        This recomputes all cache entries from scratch so that every
        transformer block sees the current sliding window context,
        making KV-cache mode equivalent to the non-KV-cache (sliding
        window recompute) path.

        A dummy noise frame is appended to x so that x.T matches the
        conditioning length (required by models like FrameDepthStackPixelDiT
        that concatenate rasterized conditioning with x along the batch
        dimension).  The dummy frame's KV cache slot is overwritten by
        the subsequent denoising loop.

        Args:
            model: Denoiser model with _reset_kv_cache_indices().
            sampler: Flow-Euler sampler (provides noise config).
            cond: Full-window conditioning dict from prepare_*_cond(use_kv_cache=False).
                  Must contain "x" key with past latent frames.
            global_start_idx: Absolute position of the first frame in x.
        """
        with torch.no_grad():
            x = cond.pop("x")
            B, Tprompt = x.shape[0], x.shape[1]
            x_shape = x.shape[2:]
            if Tprompt == 0:
                return
            # Add constant noise to past context frames
            t_cst = torch.full((B, Tprompt), sampler.cfg.min_noise_level_inference, device=x.device)
            if sampler.cfg.min_noise_level_inference > 0:
                noise = torch.randn_like(x)
                noise = torch.clamp(noise, -sampler.cfg.noise_abs_max, sampler.cfg.noise_abs_max)
                x = sampler.diffuse(x, t_cst, noise)
            # Append dummy frame so x.T matches conditioning length.
            # Use zeros (not randn) to avoid shifting the RNG state — the
            # dummy's cache slot is overwritten during decode anyway.
            dummy = torch.zeros(B, 1, *x_shape, device=x.device)
            x = torch.cat([x, dummy], dim=1)
            t_cst = torch.cat([t_cst, torch.ones(B, 1, device=x.device)], dim=1)
            model._reset_kv_cache_indices()
            cond_ = {k: v.clone() for k, v in cond.items()}
            _ = model(x, t_cst, cond_, global_start_idx, prefill=True)

    # note: override needed to make device querying/switching work
    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """

        for module in self.models.values():
            if hasattr(module, 'device'):
                return module.device
        for module in self.models.values():
            if hasattr(module, 'parameters'):
                return next(module.parameters()).device
        return torch.device("cpu")

    # note: override needed to make device querying/switching work
    def to(self, device: torch.device) -> None:
        for module in self.models.values():
            module.to(device)

        if self.voxel_sampler:
            self.voxel_sampler.device = device

        if self.pixel_sampler:
            self.pixel_sampler.device = device

        if self.latent_norm:
            for k, v in self.latent_norm.items():
                for kk, vv in v.items():
                    self.latent_norm[k][kk] = vv.to(device)

    def _clear_kv_caches(self):
        """Reset all KV and raster caches to their empty/default state.

        Must be called before warm_start when use_kv_cache=True so that
        warm_start runs in pure non-KV mode (no stale data from a
        previous episode).
        """
        if hasattr(self.voxel_denoiser, 'temporal_kv_caches'):
            self.voxel_denoiser.temporal_kv_caches = []
        if hasattr(self.voxel_denoiser, 'cross_kv_caches'):
            self.voxel_denoiser.cross_kv_caches = []
        if hasattr(self.pixel_denoiser, 'kv_caches'):
            self.pixel_denoiser.kv_caches = None
        if hasattr(self.pixel_denoiser, 'raster_cache'):
            self.pixel_denoiser.raster_cache = None

    @torch.no_grad()
    def __call__(
            self,
            context,
            num_frames,
            output_latents=True,
            output_decoded_pixels=True,
            pixel_use_x0: bool = True,
            use_camera_gt: bool = False,
            verbose: bool = True,
    ):
        raise NotImplementedError

class VoxelFirstPipeline(BasePipeline):

    def __init__(
            self,
            models: Union[Dict, nn.ModuleDict],
            accelerator=None,
            voxel_sampler=None,
            pixel_sampler=None,
            latent_normalization_stats=None,
            batch_size_voxel_vae: int = 8,
            batch_size_pixel_vae: int = 16,
            reencode_voxels: bool = True,
            wait_every: int = 200, # if accelerator is provided, how often to sync GPUs during autoregressive rollout (in number of frames)
    ):
        super().__init__(
            models,
            accelerator,
            voxel_sampler,
            pixel_sampler,
            latent_normalization_stats,
            batch_size_voxel_vae,
            batch_size_pixel_vae,
            reencode_voxels,
            wait_every,
        )

    def set_use_optimized_rasterizer(self, use_optimized: bool):
        """Enable/disable optimized rasterizer on the pixel denoiser."""
        if hasattr(self.pixel_denoiser, 'rasterizer'):
            self.pixel_denoiser.rasterizer.use_optimized = use_optimized

    def compile_blocks(self):
        """torch.compile individual transformer blocks (skips non-compilable rasterizer)."""
        for i, block in enumerate(self.pixel_denoiser.blocks):
            self.pixel_denoiser.blocks[i] = torch.compile(block)
        for i, block in enumerate(self.voxel_denoiser.blocks):
            self.voxel_denoiser.blocks[i] = torch.compile(block)

    def reset_and_warmstart_kv_caches(self, batch_size: int, rollout=None):
        """Allocate KV cache buffers for both denoisers.

        No warm-start population is done here — the prefill pass at the
        start of each frame's denoising loop repopulates the caches from
        the current sliding window, ensuring cross-layer consistency.
        """
        if hasattr(self.voxel_denoiser, '_initialize_kv_caches'):
            self.voxel_denoiser.batch_size = batch_size
            self.voxel_denoiser._initialize_kv_caches()
        if hasattr(self.pixel_denoiser, '_initialize_kv_caches'):
            self.pixel_denoiser.batch_size = batch_size
            self.pixel_denoiser._initialize_kv_caches()

    @torch.no_grad()
    def __call__(
        self,
        context,
        num_frames,
        output_latents = True,
        output_decoded_pixels = True,
        pixel_use_x0: bool = True,
        use_camera_gt: bool = False,
        keep_on_device: bool = False,
        use_kv_cache: bool = False,
        verbose: bool = True,
        time_inference: bool = False,
    ):
        """
            context: dict with starting conditions for the generation.
            By convention, we assume the context tuple is of the form (p0, c0, v0, a0:T)
                # p0 pixel observation at timestep 0
                # c0 camera at timestep 0 (or for full episode when use_camera_gt=True)
                # [optional] voxel observation at timestep t0
                # action sequence for the full episode [t0 -> num_frames]
        """

        rollout_device = self.device if keep_on_device else torch.device("cpu")

        assert "action" in context
        assert context["action"].shape[1] >= num_frames, f"Not enough actions ({context['action'].shape[1]}) to generate {num_frames} frames"

        assert "camera" in context, "Initial context must contain 'camera' key"
        B, Tc = context["camera"].shape[0], 1
        assert Tc == 1, "This pipeline only supports single-initial frame initialisation"

        rollout = {
            "action": context["action"][:, :Tc].clone().to(rollout_device),
            "camera": context["camera"][:, :Tc].clone().to(rollout_device)}
        if "pixel" in context:
            assert "pixel_latents" not in context
            rollout["pixel_latents"] = self.encode_pixels(context["pixel"]).to(rollout_device)
        elif "pixel_latents" in context:
            rollout["pixel_latents"] = context["pixel_latents"].clone().to(rollout_device)
        else:
            raise ValueError("Initial context must contain 'pixel' or 'pixel_latents'")
        assert rollout["pixel_latents"].shape[:2] == (B, Tc), "Mismatch between provided pixel and camera context"

        if "voxel" in context:
            assert "voxel_latents" not in context
            rollout["voxel"] = context["voxel"][:, :Tc].clone().to(rollout_device, dtype=torch.int16)
            rollout["voxel_latents"] = self.encode_voxels(context["voxel"]).to(rollout_device)
        elif "voxel_latents" in context:
            rollout["voxel_latents"] = context["voxel_latents"].clone().to(rollout_device)
            rollout["voxel"] = self.decode_voxels(rollout["voxel_latents"]).to(rollout_device, dtype=torch.int16)

        # Clear any stale KV/raster caches BEFORE warm_start so it runs
        # in pure non-KV mode.  Without this, episode 2+ warm_start
        # attends to episode 1's cached representations.
        if use_kv_cache:
            self._clear_kv_caches()

        rollout = self.warm_start(rollout, pixel_use_x0, rollout_device)

        if use_kv_cache:
            self.reset_and_warmstart_kv_caches(batch_size=B, rollout=rollout)

        vox_x_shape = (self.voxel_denoiser.in_channels, *self.voxel_denoiser.vox_grid_size)
        pix_x_shape = (self.pixel_denoiser.in_channels, self.pixel_denoiser.input_h, self.pixel_denoiser.input_w)

        fps_warmup_frames = 2
        frame_times = []
        times_voxel_denoise = []
        times_camera = []
        times_pix_cond = []
        times_pixel_denoise = []

        for i in tqdm(range(Tc, num_frames), desc="Autoreg rollout", disable=not verbose):
            torch.cuda.synchronize()
            _t_frame_start = time.perf_counter()
            # 0. Add next action to rollout
            rollout["action"] = torch.cat((rollout["action"], context["action"][:, i].unsqueeze(1).to(rollout_device)), dim=1)

            if time_inference:
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            #1. Diffuse next voxel frame
            if time_inference: torch.cuda.synchronize(); _t0 = time.perf_counter()
            if use_kv_cache:
                # Prefill: recompute KV cache from current window for cross-layer consistency
                prefill_cond = self.prepare_vox_denoiser_cond(rollout, i, use_kv_cache=False)
                n_ctx_vox = prefill_cond["x"].shape[1]
                self.prefill_kv_cache(self.voxel_denoiser, self.voxel_sampler, prefill_cond, global_start_idx=i - n_ctx_vox)
                # Denoise single new frame against prefilled cache
                cond_vox = self.prepare_vox_denoiser_cond(rollout, i, use_kv_cache=True)
                pred_vox, _ = self.denoise_one_frame_kv_cache(
                    self.voxel_denoiser, self.voxel_sampler,
                    cond=cond_vox, global_ts=i, x_shape=vox_x_shape,
                )
            else:
                cond_vox = self.prepare_vox_denoiser_cond(rollout, i, use_kv_cache=False)
                pred_vox, _ = self.denoise_one_frame(
                    self.voxel_denoiser, self.voxel_sampler,
                    cond=cond_vox, global_ts=i,
                )
            voxel = self.decode_voxels(pred_vox)
            if self.reencode_voxels:
                pred_vox = self.encode_voxels(voxel)
            if time_inference: torch.cuda.synchronize(); times_voxel_denoise.append(time.perf_counter() - _t0)
            rollout["voxel_latents"] = torch.cat((rollout["voxel_latents"], pred_vox.to(rollout_device)), dim=1)
            rollout["voxel"] = torch.cat((rollout["voxel"], voxel.to(rollout_device)), dim=1)

            #2. Predict next camera
            if time_inference: torch.cuda.synchronize(); _t0 = time.perf_counter()
            if not use_camera_gt:
                cond_cam = self.prepare_camera_cond(rollout, i)
                pred_cam = self.camera_model.predict_next_camera(*cond_cam)[:, -1:]  # (B, 1, cam_dim)
                rollout["camera"] = torch.cat((rollout["camera"], pred_cam.to(rollout_device)), dim=1)
            else:
                rollout["camera"] = torch.cat([rollout["camera"], context["camera"][:, i].unsqueeze(1).to(rollout_device)], dim=1)
            if time_inference: torch.cuda.synchronize(); times_camera.append(time.perf_counter() - _t0)

            #3. Pixel conditioning + prefill + denoise
            if time_inference: torch.cuda.synchronize(); _t0 = time.perf_counter()
            if use_kv_cache:
                # Prefill: recompute KV cache from current window for cross-layer consistency
                prefill_cond = self.prepare_pix_denoiser_cond(rollout, i, use_kv_cache=False)
                n_ctx_pix = prefill_cond["x"].shape[1]
                self.prefill_kv_cache(self.pixel_denoiser, self.pixel_sampler, prefill_cond, global_start_idx=(i + 1) - n_ctx_pix)
                cond_pix = self.prepare_pix_denoiser_cond(rollout, i, use_kv_cache=True)
            else:
                cond_pix = self.prepare_pix_denoiser_cond(rollout, i, use_kv_cache=False)
            if time_inference: torch.cuda.synchronize(); times_pix_cond.append(time.perf_counter() - _t0)

            #4. Diffuse next pixel frame
            if time_inference: torch.cuda.synchronize(); _t0 = time.perf_counter()
            if use_kv_cache:
                pred_pix, _ = self.denoise_one_frame_kv_cache(
                    self.pixel_denoiser, self.pixel_sampler,
                    cond=cond_pix, global_ts=i+1, x_shape=pix_x_shape,
                )
            else:
                pred_pix, _ = self.denoise_one_frame(
                    self.pixel_denoiser, self.pixel_sampler,
                    cond=cond_pix, global_ts=i+1,
                )
            if time_inference: torch.cuda.synchronize(); times_pixel_denoise.append(time.perf_counter() - _t0)
            rollout["pixel_latents"] = torch.cat((rollout["pixel_latents"], pred_pix.to(rollout_device)), dim=1)

            if time_inference:
                torch.cuda.synchronize()
                frame_times.append(time.perf_counter() - t_start)

            if self.accelerator is not None and (i % self.wait_every == 0 or i == num_frames - 1):
                self.accelerator.wait_for_everyone()

            torch.cuda.synchronize()
            frame_time = time.perf_counter() - _t_frame_start
            if (i - Tc) >= fps_warmup_frames:
                frame_times.append(frame_time)

        if frame_times:
            avg_ms = sum(frame_times) / len(frame_times) * 1000
            logger.info(f"Rollout speed (excl. first {fps_warmup_frames} frames): {1000/avg_ms:.2f} fps  ({avg_ms:.1f} ms/frame,  {len(frame_times)} frames measured)")
        if time_inference and len(times_voxel_denoise) > fps_warmup_frames:
            s = fps_warmup_frames
            def _avg_ms(lst): return sum(lst[s:]) / len(lst[s:]) * 1000
            logger.info(f"  Voxel denoise:    {_avg_ms(times_voxel_denoise):.1f} ms/frame")
            logger.info(f"  Camera predict:   {_avg_ms(times_camera):.1f} ms/frame")
            logger.info(f"  Pixel cond/proj:  {_avg_ms(times_pix_cond):.1f} ms/frame")
            logger.info(f"  Pixel denoise:    {_avg_ms(times_pixel_denoise):.1f} ms/frame")

        # prepare outputs
        for key in rollout:
            rollout[key] = rollout[key][:, 1:] #remove the ts-1 frame to prevent misalignment with GT sequence
        if output_decoded_pixels:
            rollout["pixel"] = self.decode_pixels(rollout["pixel_latents"]).to(rollout_device)
        if not output_latents:
            del rollout["pixel_latents"]
            del rollout["voxel_latents"]

        return rollout


    def warm_start(
            self,
            rollout,
            pixel_use_x0: bool,
            device,
    ):
        """Warm-start the rollout to account for different initial conditions and frame offsets between models."""

        def update_rollout_with_dummy_frame(rollout, dummy_frame, keys_added_to_rollout):
            for key in dummy_frame:
                if key not in keys_added_to_rollout:
                    rollout[key] = torch.cat((dummy_frame[key], rollout[key]), dim=1)
                    keys_added_to_rollout.append(key)
            return rollout, keys_added_to_rollout

        # add dummy frame at t-1 for all modalities
        keys_added_to_rollout = []
        dummy_frame = {
            "action" : torch.zeros_like(rollout["action"][:, :1]), # no-op action
        }
        rollout, keys_added_to_rollout = update_rollout_with_dummy_frame(rollout, dummy_frame, keys_added_to_rollout)

        if not "voxel_latents" in rollout:
            # generate vox_t0 | act_t0 + (cam_t-1, pix_t-1) (assuming static camera, cam_t-1 = cam_t0 and pix_t-1 == pix_t0)
            cond_vox = self.prepare_vox_denoiser_cond(rollout, 0)
            pred_vox, _ = self.denoise_one_frame(
                self.voxel_denoiser,
                self.voxel_sampler,
                cond=cond_vox,
                global_ts=0,
            )
            voxel = self.decode_voxels(pred_vox)
            if self.reencode_voxels:
                pred_vox = self.encode_voxels(voxel)
            rollout["voxel_latents"] = pred_vox.to(device)
            rollout["voxel"] = voxel.to(device)

        # add dummy voxel frame : static agent -> static voxel
        dummy_frame["camera"] = rollout["camera"][:, :1] # static camera
        dummy_frame["voxel_latents"] = rollout["voxel_latents"][:, :1]
        dummy_frame["voxel"] = rollout["voxel"][:, :1]
        rollout, keys_added_to_rollout = update_rollout_with_dummy_frame(rollout, dummy_frame, keys_added_to_rollout)

        # generate dummy frame from scratch if not using pixel initial frame
        if not pixel_use_x0:
            # generate pix_t-1 | (cam, vox, act)_t-1 -> we use the dummy frame storing the -1 timestep
            rollout.pop("pixel_latents", None)
            cond_pix = self.prepare_pix_denoiser_cond(dummy_frame, -1)
            pred_pix, _ = self.denoise_one_frame(
                self.pixel_denoiser,
                self.pixel_sampler,
                cond=cond_pix,
                global_ts=0,
            )
            rollout["pixel_latents"] = pred_pix.to(device) # replace pixel latents with generated frame

        # generate pix_t0 | (pix, cam, vox, act)_t-1 + (cam, vox, act)_t0
        cond_pix = self.prepare_pix_denoiser_cond(rollout, 0)
        pred_pix, _ = self.denoise_one_frame(
            self.pixel_denoiser,
            self.pixel_sampler,
            cond=cond_pix,
            global_ts=1, # from the pixel denoiser pov, we are predicting the 2nd frame of the sequence
        )
        rollout["pixel_latents"] = torch.cat((rollout["pixel_latents"], pred_pix.to(device)), dim=1)

        # rollout output should now have 2 frames for each modality representing (t-1, t0)
        return rollout

    def prepare_camera_cond(self, rollout, global_ts):

        # Check sequence integrity
        assert rollout["action"].shape[1] == global_ts + 2 # (t_{-1}, t_{0}, ..., t_{global_ts})
        assert rollout["camera"].shape[1] == global_ts + 1 # (t_{-1}, t_{0}, ..., t_{global_ts-1})
        assert rollout["pixel_latents"].shape[1] == global_ts + 1 # (t_{-1}, t_{0}, ..., t_{global_ts-1})
        assert rollout["voxel_latents"].shape[1] == global_ts + 2 #(t_{-1}, t_{0}, ..., t_{global_ts})
        assert rollout["voxel"].shape[1] == global_ts + 2 #(t_{-1}, t_{0}, ..., t_{global_ts})

        cond_len = self.camera_model.context_window_size
        shift_act = shift_vox =  1
        shift_cam = 0

        camera = rollout["camera"][:, shift_cam:][:, -cond_len:].to(self.device)
        action = rollout["action"][:, shift_act:][:, -cond_len:].to(self.device)
        voxel = rollout["voxel"][:, shift_vox:][:, -cond_len:].to(self.device, dtype=torch.long)
        # crop voxel
        vx_grid_shape = torch.tensor(voxel.shape[2:])
        cropped_vx_shape = torch.tensor(self.camera_model.voxel_cond_shape)
        crop = (vx_grid_shape - cropped_vx_shape) // 2
        voxel = voxel[...,
            crop[0]:vx_grid_shape[0] - crop[0], crop[1]:vx_grid_shape[1] - crop[1], crop[2]:vx_grid_shape[2] - crop[2]]

        CamCond = namedtuple(
            "cam_cond", "x action voxel" )
        return CamCond(camera, action, voxel)

    def prepare_vox_denoiser_cond(self, rollout, global_ts, use_kv_cache=False):

        if use_kv_cache:
            # Single-frame conditioning — past context lives in the KV cache
            cond = {
                "action": rollout["action"][:, -1:].to(self.device),
                "camera": rollout["camera"][:, -1:].to(self.device),
                "pixel_latents": rollout["pixel_latents"][:, -1:].to(self.device),
            }
            return cond

        # Check sequence integrity
        assert rollout["action"].shape[1] == global_ts + 2 # (t_{-1}, t_{0}, ..., t_{global_ts})
        assert rollout["camera"].shape[1] == global_ts + 1 # (t_{-1}, t_{0}, ..., t_{global_ts-1})
        assert rollout["pixel_latents"].shape[1] == global_ts + 1 # (t_{-1}, t_{0}, ..., t_{global_ts-1})
        if "voxel_latents" in rollout:
            assert rollout["voxel_latents"].shape[1] == global_ts + 1 #(t_{-1}, t_{0}, ..., t_{global_ts-1})

        B, C, X, Y, Z = (rollout["action"].shape[0], self.voxel_denoiser.in_channels, *self.voxel_denoiser.vox_grid_size)
        shift_vox = shift_act = 1
        shift_pix = shift_cam = 0 #start one ts earlier

        cond_len_vox = self.voxel_denoiser.context_window_size - 1
        cond_len_other = self.voxel_denoiser.context_window_size
        if "voxel_latents" in rollout:
            x = rollout["voxel_latents"][:, shift_vox:][:, -cond_len_vox:]
        else:
            x = torch.empty((B, 0, C, X, Y, Z))

        cond = {
            "x" : x.to(self.device),
            "action": rollout["action"][:, shift_act:][:, -cond_len_other:].to(self.device),
            "camera": rollout["camera"][:, shift_cam:][:, -cond_len_other:].to(self.device),
            "pixel_latents" : rollout["pixel_latents"][:, shift_pix:][:, -cond_len_other:].to(self.device),
        }
        return cond
