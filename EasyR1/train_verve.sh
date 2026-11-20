#!/bin/bash
#SBATCH --job-name=verve
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# =============================================================================
# VERVE training launcher: one script for every backbone and both modes.
# Submit from this directory:
#
#   MODEL=qwen3vl MODE=opsv \
#   TRAIN_PARQUET=/data/VERVE-Data/train_s57_opsv.parquet \
#   IMAGE_DIR=/data/VERVE-Data/images \
#   HF_HOME=/path/to/hf_cache sbatch train_verve.sh
#
#   MODEL   qwen25vl | qwen3vl | qwen35 | mimo
#   MODE    plain  (accuracy reward, perception.jinja)
#           opsv   (verify reward, perception_verify2_1.jinja)
#   ANNEAL  "START END" (opsv only): retire the verification terms linearly
#           over these steps, leaving a pure accuracy reward after END.
#           Default: "40 80" for qwen35, off for the other backbones.
#
# 7 nodes x 4 GPUs = 24 training GPUs + 1 judge node. RESUME=1 resubmits this
# script at the walltime limit and continues from the last checkpoint.
# =============================================================================
set -euo pipefail

: "${MODEL:?set MODEL to qwen25vl | qwen3vl | qwen35 | mimo}"
: "${MODE:?set MODE to plain | opsv}"
: "${TRAIN_PARQUET:?set TRAIN_PARQUET}" "${IMAGE_DIR:?set IMAGE_DIR}"
export TRAIN_PARQUET IMAGE_DIR

export USE_WANDB="${USE_WANDB:-1}"
export RESUME="${RESUME:-1}"
export RESUBMIT_SCRIPT="${SLURM_SUBMIT_DIR}/train_verve.sh"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
export JUDGE_SHARDS_PER_NODE="${JUDGE_SHARDS_PER_NODE:-4}"   # MoE judge: one TP=1 server per GPU
export N_JUDGE_NODES="${N_JUDGE_NODES:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export ROLLOUT_BS="${ROLLOUT_BS:-384}"
export GLOBAL_BS="${GLOBAL_BS:-96}"
export MINI_BS="${MINI_BS:-96}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export SAVE_LIMIT="${SAVE_LIMIT:-10}"
export RAY_memory_usage_threshold=0.97
export EXP_TAG="${EXP_TAG:-verve_${MODE}}"

EXTRA_ARGS=(worker.rollout.gpu_memory_utilization=0.55)

# ----------------------------- backbone --------------------------------------
case "${MODEL}" in
  qwen25vl) MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" ;;
  qwen3vl)  MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}" ;;
  mimo)     MODEL_PATH="${MODEL_PATH:-XiaomiMiMo/MiMo-VL-7B-SFT-2508}" ;;
  qwen35)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
    ANNEAL="${ANNEAL-40 80}"
    EXTRA_ARGS+=(
      # 248k vocabulary: bound the log-prob pass to one sample-budget per device
      worker.actor.micro_batch_size_per_device_for_experience=1
      # render the chat template with thinking disabled
      data.override_chat_template="${SLURM_SUBMIT_DIR}/examples/format_prompt/qwen3_5_nothink_chat_template.jinja"
      worker.rollout.extra_engine_kwargs.gdn_prefill_backend=triton
    )
    ;;
  *) echo "[error] unknown MODEL=${MODEL}" >&2; exit 1 ;;
esac
export MODEL_PATH

# ------------------------------- mode -----------------------------------------
case "${MODE}" in
  plain) ;;  # engine defaults: perception_multinode.py reward, perception.jinja prompt
  opsv)
    export REWARD_FN="./examples/reward_function/perception_verify.py:compute_score"
    EXTRA_ARGS+=(
      data.format_prompt=./examples/format_prompt/perception_verify2_1.jinja
      worker.reward.reward_function_kwargs.reward_shape=gated_mult
      worker.reward.reward_function_kwargs.alpha=0.5
      worker.reward.reward_function_kwargs.w_fmt=0.1
      worker.reward.reward_function_kwargs.matcher=judge
    )
    if [ -n "${ANNEAL:-}" ]; then
      read -r ANNEAL_START ANNEAL_END <<<"${ANNEAL}"
      EXTRA_ARGS+=(
        worker.reward.reward_function_kwargs.anneal_start="${ANNEAL_START}"
        worker.reward.reward_function_kwargs.anneal_end="${ANNEAL_END}"
      )
    fi
    # MiMo-VL wraps every response in <think>...</think>, which the format check bans
    if [ "${MODEL}" = "mimo" ]; then
      EXTRA_ARGS+=(worker.reward.reward_function_kwargs.strip_think=true)
    fi
    ;;
  *) echo "[error] unknown MODE=${MODE}" >&2; exit 1 ;;
esac

cd "${SLURM_SUBMIT_DIR}"
source "${SLURM_SUBMIT_DIR}/train_common.sh"
