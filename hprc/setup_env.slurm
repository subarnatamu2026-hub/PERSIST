#!/bin/bash
# Batch job: build the Python venv (torch + craftium) inside the container.
# Detached from your terminal - survives logout/laptop-close.
#SBATCH --job-name=craftium-setup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
### account/partition are supplied by submit_all.sh, or uncomment & set here:
### #SBATCH --account=YOUR_ACCOUNT
### #SBATCH --partition=YOUR_PARTITION

set -e
mkdir -p logs
cd "$SCRATCH/PERSIST"
source hprc/container.sh
module load WebProxy 2>/dev/null || true    # outbound internet for uv/pip downloads

ctr_exec "cd '$SCRATCH/PERSIST' && bash hprc/setup_env.sh"
