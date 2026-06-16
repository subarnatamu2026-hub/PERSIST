import os
import random

import numpy as np
import torch
from accelerate.utils import is_xpu_available, is_mlu_available, is_sdaa_available, is_musa_available, \
    is_torch_xla_available
from loguru import logger


def accelerator_load_random_state(ckpt_dir: str, accelerator, process_index: int = 0):
    """ Load random states from a file and set the random seeds for various libraries.
        No idea if this will work for other process_index than 0.
    """

    try:
        states = torch.load(os.path.join(ckpt_dir, f"random_states_{process_index}.pkl"), weights_only=False)
        if "step" in states:
            accelerator.step = states["step"]
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        if is_xpu_available():
            torch.xpu.set_rng_state_all(states["torch_xpu_manual_seed"])
        if is_mlu_available():
            torch.mlu.set_rng_state_all(states["torch_mlu_manual_seed"])
        elif is_sdaa_available():
            torch.sdaa.set_rng_state_all(states["torch_sdaa_manual_seed"])
        elif is_musa_available():
            torch.musa.set_rng_state_all(states["torch_musa_manual_seed"])
        else:
            torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        if is_torch_xla_available():
            import torch_xla.core.xla_model as xm
            xm.set_rng_state(states["xm_seed"])
    except Exception as e:
        logger.error(f"Encountered an issue when loading the accelerator random state. Exception raised: {e}")