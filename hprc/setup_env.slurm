#!/bin/bash
# Batch job: build the Python venv (torch + craftium) inside the container.
# Detached from your terminal - survives logout/laptop-close.
#SBATCH --job-name=craftium-setup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
### account/partition are supplied by submit_all.sh, or uncomment & set here:
### #SBATCH --account=YOUR_ACCOUNT
### #SBATCH --partition=YOUR_PARTITION

set -e
mkdir -p logs
module load WebProxy 2>/dev/null || true   # outbound internet for uv/pip downloads
SIF="${SIF:-$SCRATCH/craftium.sif}"
cd "$SCRATCH/PERSIST"
singularity exec --bind /scratch "$SIF" bash hprc/setup_env.sh
