#!/bin/bash
# Batch job: build the Charliecloud image on FASTER (unprivileged) and pack it
# to a squashfs. One-time; detached. Only needed for CONTAINER_KIND=charliecloud.
#SBATCH --job-name=craftium-image
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -e
mkdir -p logs
module load "${CONTAINER_MODULE:-charliecloud}" 2>/dev/null || true
module load WebProxy 2>/dev/null || true            # outbound internet for apt/git/curl
export CH_IMAGE_STORAGE="${CH_IMAGE_STORAGE:-$SCRATCH/ch-storage}"
cd "$SCRATCH/PERSIST"

# --force lets ch-image emulate the privileged operations distro package
# managers (apt) expect inside an unprivileged build.
ch-image build --force -t craftium -f hprc/Dockerfile hprc
ch-convert craftium "${CH_IMAGE:-$SCRATCH/craftium.sqfs}"
echo "Built ${CH_IMAGE:-$SCRATCH/craftium.sqfs}"
