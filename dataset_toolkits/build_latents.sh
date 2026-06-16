#!/bin/bash

# build_latents.sh - Sequential pipeline for encoding both pixel and voxel latents

print_usage() {
    echo "Usage: $0 <dataset_directory> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --venv-path PATH            Path to uv virtualenv dir (default: .venv)"
    echo "  --num-cpus N                Number of CPU workers per rank (default: 2)"
    echo "  --pixel-vae-path PATH       Path to pixel VAE checkpoint"
    echo "  --voxel-vae-path PATH       Path to voxel VAE checkpoint"
    echo "  --batch-size-pix N          Batch size for pixel encoding (default: 16)"
    echo "  --batch-size-vox N          Batch size for voxel encoding (default: 4)"
    echo "  --mixed-precision MODE      Autocast mode for encoding: fp16, bf16, or no (default: bf16)"
    echo "  --gpus SEQUENCE             Comma-separated GPU IDs (e.g., 0,1,2,3) for parallel processing"
    echo ""
    echo "Examples:"
    echo "  $0 datasets/200ts_full --pixel-vae-path model.safetensors --voxel-vae-path model.safetensors"
    echo "  $0 datasets/200ts_full --pixel-vae-path model.safetensors --gpus 0,1,2,3"
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

OUTPUT_DIR="$1"
shift

# Default values
VENV_PATH=".venv"
NUM_CPUS=2
PIXEL_VAE_PATH=""
VOXEL_VAE_PATH=""
VOXEL_CHANNELS="32,128,512"
VOXEL_LATENT_CHANNELS=48
BATCH_SIZE_PIX=16
BATCH_SIZE_VOX=4
MIXED_PRECISION="bf16"
GPUS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --num-cpus) NUM_CPUS="$2"; shift 2 ;;
        --pixel-vae-path) PIXEL_VAE_PATH="$2"; shift 2 ;;
        --voxel-vae-path) VOXEL_VAE_PATH="$2"; shift 2 ;;
        --voxel-channels) VOXEL_CHANNELS="$2"; shift 2 ;;
        --voxel-latent-channels) VOXEL_LATENT_CHANNELS="$2"; shift 2 ;;
        --batch-size-pix) BATCH_SIZE_PIX="$2"; shift 2 ;;
        --batch-size-vox) BATCH_SIZE_VOX="$2"; shift 2 ;;
        --mixed-precision) MIXED_PRECISION="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown argument: $1"; print_usage; exit 1 ;;
    esac
done

if [ ! -d "$OUTPUT_DIR" ] || [ ! -f "$OUTPUT_DIR/metadata.csv" ]; then
    echo "Error: Invalid dataset directory '$OUTPUT_DIR'"
    exit 1
fi

# Activate the uv virtual environment
ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "Error: Could not find virtualenv activate script at '$ACTIVATE_SCRIPT'"
    exit 1
fi

echo "Activating uv environment: $VENV_PATH"
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment '$VENV_PATH'"
    exit 1
fi

# Parse GPU list and setup parallel execution
if [[ -n "$GPUS" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    WORLD_SIZE=${#GPU_ARRAY[@]}
    echo "Using ${WORLD_SIZE} GPUs: ${GPUS}"
else
    GPU_ARRAY=(0)
    WORLD_SIZE=1
    echo "Using single GPU mode (default GPU 0)"
fi

# Function to run parallel jobs with proper cleanup
run_parallel_jobs() {
    local script_path=$1
    local job_type=$2
    local batch_size=$3
    local extra_args="$4"
    
    echo "Starting parallel ${job_type} encoding on ${WORLD_SIZE} GPU(s)..."
    
    # Array to store PIDs of background processes
    local pids=()
    
    # Launch parallel jobs
    for i in "${!GPU_ARRAY[@]}"; do
        local gpu_id=${GPU_ARRAY[$i]}
        echo "Launching ${job_type} job ${i} on GPU ${gpu_id}"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python $script_path \
            --output_dir "$OUTPUT_DIR" \
            --max_workers "$NUM_CPUS" \
            --batch_size "$batch_size" \
            --mixed_precision "$MIXED_PRECISION" \
            --rank $i \
            --world_size $WORLD_SIZE \
            $extra_args &
        
        pids+=($!)
    done
    
    # Function to cleanup on exit
    cleanup() {
        echo "Terminating ${job_type} jobs..."
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
        wait
        exit 1
    }
    
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM
    
    # Wait for all jobs to complete
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "${job_type} job with PID $pid failed"
            failed=1
        fi
    done
    
    # Reset trap
    trap - SIGINT SIGTERM
    
    if [ $failed -eq 1 ]; then
        echo "${job_type} encoding failed"
        exit 1
    fi
    
    echo "${job_type} encoding completed successfully"
}

echo "Processing dataset: $OUTPUT_DIR"

# Encode voxel latents
if [[ -n "$VOXEL_VAE_PATH" ]]; then
    run_parallel_jobs \
        "dataset_toolkits/encode_voxel_latents.py" \
        "voxel" \
        "$BATCH_SIZE_VOX" \
"--model_path $VOXEL_VAE_PATH --channels $VOXEL_CHANNELS --latent_channels $VOXEL_LATENT_CHANNELS"
    
    python dataset_toolkits/build_metadata.py --output_dir "$OUTPUT_DIR" --from_file
    [ $? -ne 0 ] && echo "Metadata update failed" && exit 1
fi

# Encode pixel latents
if [[ -n "$PIXEL_VAE_PATH" ]]; then
    run_parallel_jobs \
        "dataset_toolkits/encode_pixel_latents.py" \
        "pixel" \
        "$BATCH_SIZE_PIX" \
"--model_path $PIXEL_VAE_PATH"
    
    python dataset_toolkits/build_metadata.py --output_dir "$OUTPUT_DIR" --from_file
    [ $? -ne 0 ] && echo "Metadata update failed" && exit 1
fi

# Final metadata update
python dataset_toolkits/build_metadata.py --output_dir "$OUTPUT_DIR" --from_file

echo "Latent encoding completed."