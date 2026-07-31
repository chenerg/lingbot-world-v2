#!/bin/bash
# Multi-segment prompt inference (causal_fast) on one GPU.
# Usage: bash run_multi_prompt_single_gpu.sh <weights_dir> <gpu_id>
# A high-memory GPU is required; 80 GB is recommended for this configuration.

set -euo pipefail
set -x

WEIGHT_DIR=${1:-lingbot-world-v2-14b-causal-fast}
GPU_ID=${2:-0}

CUDA_VISIBLE_DEVICES="${GPU_ID}" python generate.py \
  --task i2v-A14B \
  --size '480*832' \
  --ckpt_dir "${WEIGHT_DIR}" \
  --image examples/06/image.jpg \
  --action_path examples/06 \
  --ulysses_size 1 \
  --t5_cpu \
  --offload_model true \
  --local_attn_size 18 \
  --sink_size 6 \
  --prompts "Cover the red and blue balls with the two white cups.|||Lift the cup covering the blue ball to reveal it. Keep the red ball covered." \
  --segment_frames 81,81
