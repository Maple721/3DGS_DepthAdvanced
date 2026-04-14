#!/bin/bash

# Render images from custom camera poses
# Usage: ./render.sh --model_path /path/to/dataset --pose_file /path/to/poses.json [--iteration N]

# Default values
MODEL_PATH=""
POSE_FILE=""
ITERATION=-1
GPU_ID=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --pose_file)
            POSE_FILE="$2"
            shift 2
            ;;
        --iteration)
            ITERATION="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Usage: ./render.sh --model_path /path/to/dataset --pose_file /path/to/poses.json [--iteration N] [--gpu ID]"
    exit 1
fi

if [ -z "$POSE_FILE" ]; then
    echo "Error: --pose_file is required"
    echo "Usage: ./render.sh --model_path /path/to/dataset --pose_file /path/to/poses.json [--iteration N] [--gpu ID]"
    exit 1
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Generate output directory based on dataset name and pose file name
DATASET_NAME=$(basename "$MODEL_PATH")
POSE_NAME=$(basename "$POSE_FILE" .json)
OUTPUT_DIR="output/${DATASET_NAME}"

echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Pose file:  $POSE_FILE"
echo "Output:     $OUTPUT_DIR"
echo "GPU:        $GPU_ID"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run rendering
python render_custom_pose.py \
    --model_path "output/${MODEL_PATH}" \
    --pose_file "$POSE_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --iteration "$ITERATION" \
    "$@"

echo ""
echo "Rendering complete! Results saved to: $OUTPUT_DIR"
echo "  RGB images:   $OUTPUT_DIR/rgb/"
echo "  Depth maps:   $OUTPUT_DIR/depth/"
