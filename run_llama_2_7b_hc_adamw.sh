#!/bin/bash

# Llama 2 7B pretraining with AdamW and the original hyper-connections (hc):
#   hc   H_pre / H_res / H_post = static + scale * tanh(dynamic), unconstrained
#        (same as --hyper_conn_type=hc in MHC-lite-exploration)
# All other settings come from train_configs/llama2_7b.toml and the hyper-connection
# defaults (n = 4 streams, reduce = sum).
#
# H_res logging (SAVE_H_RES_FREQ) only applies to identity_tanh_offdiag, so it does nothing for hc.
#
# Usage:
#   ./run_llama_2_7b_hc_adamw.sh
#   NGPU=8 ./run_llama_2_7b_hc_adamw.sh
#   SAVE_H_RES_FREQ=50 ./run_llama_2_7b_hc_adamw.sh  # save H_res every 50 steps
#   ./run_llama_2_7b_hc_adamw.sh --training.steps 100   # extra args are passed to train.py

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# H_res logging frequency (default: every 100 steps)
SAVE_H_RES_FREQ=${SAVE_H_RES_FREQ:-"100"}

# pure fp32 training, also with NGPU > 1 (FSDP would otherwise use bf16 mixed precision)
METHODS="hc" \
    SAVE_H_RES_FREQ=${SAVE_H_RES_FREQ} \
    ./run_llama_2_7b_hc.sh --optimizer.name AdamW \
    --training.mixed_precision_param float32 \
    "$@"
