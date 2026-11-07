#!/bin/bash

# Llama 2 7B pretraining with hyper-connection residuals.
# Runs every method in METHODS one after another.
#
# Usage:
#   ./run_llama_2_7b_hc.sh
#   NGPU=8 METHODS="mhc mhc_lite" HC_N=4 ./run_llama_2_7b_hc.sh
#   METHODS="identity_tanh_offdiag" ./run_llama_2_7b_hc.sh --model.mhc_h_res_offdiag_fixed
#   ./run_llama_2_7b_hc.sh --training.steps 100   # extra args are passed to train.py
#
# The global batch is fixed by GLOBAL_BATCH_SIZE (sequences per optimizer step), independent of NGPU:
#   grad_accumulation_steps = GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * NGPU)
#   tokens per step         = GLOBAL_BATCH_SIZE * seq_len (2048 in llama2_7b.toml)
# e.g. NGPU=8 GLOBAL_BATCH_SIZE=32 MICRO_BATCH_SIZE=2 -> grad_accumulation_steps=2.
# Set the batch through these variables, not --training.batch_size / --training.grad_accumulation_steps.
#
# Data: the training data order and the validation sharding follow a DATA_DP_DEGREE-GPU run, whatever
# NGPU is (--training.emulate_dp_degree). The default 2 makes 1-GPU runs see the same data as 2-GPU runs;
# each GPU then alternates between the data of DATA_DP_DEGREE / NGPU ranks over the micro-batches.
# Validation uses VAL_MICRO_BATCH_SIZE (default MICRO_BATCH_SIZE); set it to the micro batch of the
# run to match, e.g. VAL_MICRO_BATCH_SIZE=2 to match a 2-GPU run with MICRO_BATCH_SIZE=2.
#
# Precision: --training.mixed_precision_param float32 trains in pure fp32 on any number of GPUs,
# matching single-GPU runs (FSDP would otherwise use bf16 mixed precision when NGPU > 1).

# conda activate torchtitan

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1

export WANDB_API_KEY="${WANDB_API_KEY:?please set WANDB_API_KEY}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

NGPU=${NGPU:-"1"}
NNODES=${NNODES:-"1"}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama2_7b.toml"}

# none, hc, mhc, mhc_lite, identity, identity_tanh_offdiag
METHODS=${METHODS:-"identity_tanh_offdiag identity mhc mhc_lite"}
HC_N=${HC_N:-"4"}
HC_REDUCE_MODE=${HC_REDUCE_MODE:-"sum"}
HC_OFFDIAG_INIT_SCALE=${HC_OFFDIAG_INIT_SCALE:-"0.001"}

# global batch (sequences per optimizer step) and per-GPU micro batch
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-"8"}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-"2"}
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * NGPU) != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} x NGPU=${NGPU}" >&2
    exit 1
fi
GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * NGPU) ))
echo "global batch: ${GLOBAL_BATCH_SIZE} seqs = ${MICRO_BATCH_SIZE} (micro) x ${NGPU} (gpus) x ${GRAD_ACCUM_STEPS} (grad accum)"

# data order and validation sharding of a DATA_DP_DEGREE-GPU run
DATA_DP_DEGREE=${DATA_DP_DEGREE:-"2"}
VAL_MICRO_BATCH_SIZE=${VAL_MICRO_BATCH_SIZE:-${MICRO_BATCH_SIZE}}
if (( DATA_DP_DEGREE % NGPU != 0 )); then
    echo "DATA_DP_DEGREE=${DATA_DP_DEGREE} must be a multiple of NGPU=${NGPU}, e.g. DATA_DP_DEGREE=${NGPU}" >&2
    exit 1
fi
if (( GRAD_ACCUM_STEPS % (DATA_DP_DEGREE / NGPU) != 0 )); then
    echo "grad accum steps ${GRAD_ACCUM_STEPS} must be a multiple of DATA_DP_DEGREE / NGPU = $(( DATA_DP_DEGREE / NGPU ))" >&2
    exit 1
fi
echo "data: as on ${DATA_DP_DEGREE} gpus, validation micro batch ${VAL_MICRO_BATCH_SIZE}"

# checkpoint: model weights only, saved once at the last step (training.steps), in DCP format under
#   outputs/checkpoint/<model>_<flavor>/<optimizer><hc tag>/step-<steps>
# the huge interval disables intermediate (full-state) saves

# H_res logging: save H_res matrices every N steps (0 to disable)
SAVE_H_RES_FREQ=${SAVE_H_RES_FREQ:-"100"}

for method in $METHODS; do
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    train.py --job.config_file ${CONFIG_FILE} \
    --model.hyper_conn_type ${method} \
    --model.hyper_conn_n ${HC_N} \
    --model.hyper_conn_reduce_mode ${HC_REDUCE_MODE} \
    --model.mhc_h_res_offdiag_init_scale ${HC_OFFDIAG_INIT_SCALE} \
    --model.save_h_res_freq ${SAVE_H_RES_FREQ} \
    --training.batch_size ${MICRO_BATCH_SIZE} \
    --training.grad_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --training.emulate_dp_degree ${DATA_DP_DEGREE} \
    --training.val_batch_size ${VAL_MICRO_BATCH_SIZE} \
    --training.mixed_precision_param float32 \
    --checkpoint.enable_checkpoint \
    --checkpoint.model_weights_only \
    --checkpoint.folder checkpoint \
    --checkpoint.interval 1000000000 \
    "$@"
done
