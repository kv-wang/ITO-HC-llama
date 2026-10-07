#!/bin/bash

# Llama 2 7B pretraining with AdamW, one run per hyper-connection method:
#   identity               H_res = I
#   mhc                    H_res = Sinkhorn(...)
#   identity_tanh_offdiag  H_res = I + gamma * tanh(offdiag), gamma trainable
# All other settings come from train_configs/llama2_7b.toml and the hyper-connection
# defaults (n = 4 streams, reduce = sum, gamma init 0.05).
#
# Usage:
#   ./run_llama_2_7b_hc_adamw.sh
#   NGPU=8 ./run_llama_2_7b_hc_adamw.sh
#   ./run_llama_2_7b_hc_adamw.sh --training.steps 100   # extra args are passed to train.py

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# pure fp32 training, also with NGPU > 1 (FSDP would otherwise use bf16 mixed precision)
METHODS="identity_tanh_offdiag identity mhc" \
    ./run_llama_2_7b_hc.sh --optimizer.name AdamW \
    --training.mixed_precision_param float32 \
    "$@"
