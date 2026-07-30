#!/bin/bash
# Multi-segment prompt inference (causal_fast) on N GPUs.
# Usage: bash run_multi_prompt.sh <weights_dir> <num_gpus>
# num_gpus must divide 40 (heads): 2/4/5/8/...

set -x

WEIGHT_DIR=${1:-lingbot-world-v2-14b-causal-fast}
NGPU=${2:-8}

torchrun --nproc_per_node=${NGPU} generate.py \
  --task i2v-A14B \
  --size 480*832 \
  --ckpt_dir ${WEIGHT_DIR} \
  --image examples/06/image.jpg \
  --action_path examples/06 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size ${NGPU} \
  --local_attn_size 18 \
  --sink_size 6 \
  --prompts "Cover the red and blue balls with the two white cups.|||Lift the cup covering the blue ball to reveal it. Keep the red ball covered." \
  --segment_frames 81,81
