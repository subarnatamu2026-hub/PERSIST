#!/bin/bash
# =====================================================================
# Upload the generated datasets to Hugging Face as a detached SLURM job
# (survives logout). Submit AFTER grace_generate.slurm has finished.
#
#   cd $SCRATCH/PERSIST
#   sbatch --account=<your_account> --partition=medium \
#     --export=ALL,HF_TOKEN=hf_xxx,HF_REPO=<user>/<dataset> \
#     hprc/push_hf.slurm
#
# (add HF_PRIVATE=1 to the --export list to make the repo private)
# =====================================================================
#SBATCH --job-name=hf-push
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e
cd "$SCRATCH/PERSIST"
mkdir -p logs
bash hprc/push_to_hf.sh
