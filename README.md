# World Models with Persistent 3D State

<p>
  <a href="https://francelico.github.io/persist.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2603.03482"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv Paper"></a>
  <a href="https://huggingface.co/PERSIST-team"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Models"></a>
  <a href="https://x.com/SamuelGarcin/status/2029579939431715212?s=20"><img src="https://img.shields.io/badge/X-Thread-black?logo=x" alt="X Thread"></a>
</p>

Official codebase for [PERSIST](https://francelico.github.io/persist.github.io/) (ICML 2026), a world model that generates coherent rollouts over thousands of steps by modelling a dynamic 3D state of the world.
<div align="center">
<img width="720" alt="hero_loop90" src="https://github.com/user-attachments/assets/adbe3f57-4f62-4418-a767-6ebb64a68ba8" />
</div>

## PERSIST
PERSIST is a voxel-based latent-diffusion **world model**. Given an
initial image and camera pose, PERSIST generates an **interactive**, **dynamic** and **explorable** voxel world.

<img width="2500" height="993" alt="image" src="https://github.com/user-attachments/assets/ed2d2a9f-7eae-40b9-9284-e6c9afc1eb9f" />

PERSIST autoregressively predicts future
voxel world states, pixel observations and camera states based on input player actions. Generation runs as a multi-stage pipeline: the
voxel denoiser predicts the next voxel latent, the camera model predicts the next pose, and the pixel
denoiser predicts the next pixel observation. Two VAEs
map voxels/pixels to and from latent space.

## Installation

> [!WARNING]  
> We strongly recommend following this installation guide when running inference on our pre-trained models. A mismatch in CUDA or package versions from the original training setup can cause generation artifacts. Installation instructions were tested for Ubuntu 22.04.

### Pre-requisites

**1. Install System libraries** (Ubuntu 22.04):
```bash
sudo apt install g++ make libc6-dev cmake fontconfig libpng-dev libjpeg-dev libgl1-mesa-dev \
  libgles2-mesa-dev libsqlite3-dev libglfw3-dev libogg-dev libvorbis-dev libopenal-dev \
  libcurl4-gnutls-dev libfreetype6-dev zlib1g-dev libgmp-dev libjsoncpp-dev libzstd-dev \
  libluajit-5.1-dev gettext libsdl2-dev libpython3-dev xvfb
```

**2. Install CUDA 12.6**
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
```

If CUDA 12.6 is already installed (e.g. via conda), make sure `CUDA_HOME` is set to point to it.

**3. Install uv**

We recommend [`uv`](https://docs.astral.sh/uv/) to manage python dependencies. 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Python environment

**1. Clone and install:**
```bash
git clone https://github.com/francelico/PERSIST
cd PERSIST
git submodule update --init --recursive

# Inference + training CUDA build
uv sync --group cu
source .venv/bin/activate

# install nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
cd /tmp/extensions/nvdiffrast
git checkout tags/v0.3.3
uv pip install /tmp/extensions/nvdiffrast --no-build-isolation
```

**2. [OPTIONAL] Install Craftium dependencies to generate new datasets**

> [!NOTE]  
> `run_inference.py` downloads a sample dataset from HuggingFace. This step is only necessary for generating new datasets using craftium. 

```bash

# build SDL2 from source
cd /tmp \
  && git clone https://github.com/libsdl-org/SDL.git SDL2 \
  && cd SDL2 \
  && git checkout release-2.28.5 \
  && mkdir -p build \
  && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
  && make -j$(nproc) \
  && make install \
  && ldconfig \
  && cd /tmp \
  && rm -rf SDL2
  
uv sync --group env
```

## Inference

The quickest way to run inference is to let the pretrained weights and a sample evaluation
dataset download automatically from the [`PERSIST-team`](https://huggingface.co/PERSIST-team) HuggingFace hub:

```bash
uv run python -m scripts.run_inference \
  --pipeline-variant S \
  --checkpoints-namespace PERSIST-team \
  --dataset-repo PERSIST-team/persist-eval-sample \
  --num-frames 256 \
  --output-dir outputs/rollouts
```

> [!NOTE]  
> You can run multi-GPU inference with
> 
> `uv run accelerate launch --multi_gpu --mixed_precision bf16 --num_processes <N_GPU> -m scripts.run_inference ...`

### Inference configurations
| Configuration | Command line arguments |
|---|---|
| PERSIST-S | `--pipeline-variant S` |
| PERSIST-XL | `--pipeline-variant XL` |
| PERSIST-S/XL $+w_0$ (voxel initialization)| `--include-initial-voxel-frame --include-initial-pixel-frame` |

## Training

Each model in the pipeline has its own training script. Launch with `accelerate`:

```bash
uv run accelerate launch --multi_gpu --mixed_precision bf16 --num_processes <N_GPU> \
  train_scripts/<script>.py <args...>
```

| Script | Model |
|---|---|
| `train_scripts/train_vae_voxel.py` | voxel VAE (`ResNet3dEncoder`/`ResNet3dDecoder`) |
| `train_scripts/train_vae_pixel.py` | pixel VAE (`ViTVae`) |
| `train_scripts/train_diffuser_voxel.py` | voxel denoiser (`VoxelDiT`) |
| `train_scripts/train_diffuser_pixel.py` | pixel denoiser (`FrameDepthStackPixelDiT`) |
| `train_scripts/train_transformer_camera.py` | camera model (`CameraTransformer`) |

The diffuser scripts train in latent space and expect
a dataset of pre-encoded latents (see below).

## Dataset generation and processing

### Generate a new dataset
This step requires installing the uv [env] group to generate data with Craftium (see installation instructions).
```bash
# 1. Generate dataset_params.json
uv run python dataset_toolkits/generate_raw_data.py --dataset_dir datasets --dataset_name mydata \
  --env_id OpenWorldCreative-v0 --ep_timesteps 400 --init --num_levels 10000
# 2. Spawn a data generating process. Use --world_size and --rank to split data generation across multiple processes.
#    Each $RANK generates $NUM_LEVELS // $WORLD_SIZE episodes.
uv run python dataset_toolkits/generate_raw_data.py --dataset_dir datasets --dataset_name mydata \
  --env_id OpenWorldCreative-v0 --disable_commit_check --ep_timesteps 400 --world-size 10 --rank $RANK
```

### Building the dataset

Run the following script to finish building the dataset. The resulting dataset lets us train the pixel and voxel VAEs.
```bash
bash dataset_toolkits/build_dataset.sh datasets/mydata
```

Pre-encode pixel and voxel latents. This step is necessary to train the latent denoisers. It requires access to pre-trained pixel and voxel VAE checkpoints.
```
# 1. Encode voxel/pixel latents with trained VAEs
bash dataset_toolkits/build_latents.sh datasets/mydata \
  --voxel-vae-path path/to/voxel_vae_encoder.safetensors \
  --pixel-vae-path path/to/pixel_vae.safetensors

# 2. Compute latent normalization statistics
uv run python dataset_toolkits/compute_latent_stats.py --dataset_path datasets/mydata \
  --process_pixel --process_voxel
```

The structure and dataset layout is documented in `dataset_toolkits/data.md`.

## Cite us

```
@inproceedings{garcin2026beyond,
  title={Beyond Pixel Histories: World Models with Persistent 3D State},
  author={Garcin, Samuel and Walker, Thomas and McDonagh, Steven and Pearce, Tim and Bilen, Hakan and He, Tianyu and Wang, Kaixin and Bian, Jiang},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
