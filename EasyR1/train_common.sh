# =============================================================================
# SLURM training engine for VERVE (multi-node GRPO with a dedicated judge node).
#
# SOURCED at the end of a launcher (train_verve.sh), NOT run directly. The
# launcher owns the #SBATCH header and sets the run knobs below.
#
# Topology (written for nodes with 4 GPUs each):
#   nodes[0 .. N-1-J]  -> verl training (Ray head on nodes[0]), all GPUs each
#   last J nodes       -> judge vLLM servers (J = N_JUDGE_NODES, default 1)
# e.g. --nodes=7, N_JUDGE_NODES=1 => 24 training GPUs + 1 judge node.
# Rollout/global batch sizes must divide evenly by the training world size
# (N_GPUS_PER_NODE * training nodes).
#
# Compute nodes are assumed to be OFFLINE: models must already sit in HF_HOME
# and W&B runs offline (`wandb sync` later).
#
# Inputs the launcher MUST set: MODEL_PATH, JUDGE_MODEL, TRAIN_PARQUET,
#   IMAGE_DIR, EXP_TAG, HF_HOME
# Optional (safe defaults): EASYR1_DIR (default: the sbatch submit directory),
#   VENV, JUDGE_VENV, EXP_NAME, USE_WANDB, RESUME, N_JUDGE_NODES,
#   RESUBMIT_SCRIPT (required if RESUME=1), N_GPUS_PER_NODE, ROLLOUT_BS,
#   GLOBAL_BS, MINI_BS, TOTAL_EPOCHS, MAX_STEPS, SAVE_FREQ, SAVE_LIMIT,
#   REWARD_FN, EXTRA_ARGS (bash array of config overrides)
# =============================================================================

set -euo pipefail
echo "start training"

# This script only works inside a SLURM allocation (it discovers its nodes
# from SLURM_JOB_NODELIST). Submit the launcher with sbatch, don't run bash.
if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "[error] not inside a SLURM job (SLURM_JOB_NODELIST unset)." >&2
  echo "[error] Submit with:  sbatch train_verve.sh" >&2
  exit 1
fi

# sbatch copies the launcher to its spool directory, so the tree is anchored on
# the submit directory rather than on ${BASH_SOURCE[0]}.
EASYR1_DIR="${EASYR1_DIR:-${SLURM_SUBMIT_DIR}}"
VENV="${VENV:-${EASYR1_DIR}/.venv}"
JUDGE_VENV="${JUDGE_VENV:-${VENV}}"   # the judge servers may run from another environment

# Models and data are owned by the LAUNCHER — no
# defaults here. MODEL_PATH/JUDGE_MODEL are HF ids or local dirs, resolved
# OFFLINE from the local cache (HF_HOME below).
: "${MODEL_PATH:?launcher must set MODEL_PATH before sourcing train_common.sh}"
: "${JUDGE_MODEL:?launcher must set JUDGE_MODEL before sourcing train_common.sh}"
: "${TRAIN_PARQUET:?launcher must set TRAIN_PARQUET before sourcing train_common.sh}"
: "${IMAGE_DIR:?launcher must set IMAGE_DIR before sourcing train_common.sh}"
: "${EXP_TAG:?launcher must set EXP_TAG before sourcing train_common.sh}"

# GPU carve (4 GPUs per node). Training nodes give all 4
# GPUs to Ray; the dedicated judge node gives all 4 GPUs to one vLLM server.
TRAIN_GPUS="0,1,2,3"
JUDGE_GPUS="0,1,2,3"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"   # training GPUs per node (divisible by rollout TP=2)
RAY_PORT=6379                             # Ray GCS (head --port / worker --address)
JUDGE_PORT=8270

# --- Run knobs (launcher may override; sensible defaults otherwise) ----------
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"       # save every N steps (-1 disables saving)
SAVE_LIMIT="${SAVE_LIMIT:-1}"      # keep at most N checkpoints on disk
ROLLOUT_BS="${ROLLOUT_BS:-384}"
GLOBAL_BS="${GLOBAL_BS:-96}"
MINI_BS="${MINI_BS:-96}"
declare -p EXTRA_ARGS >/dev/null 2>&1 || EXTRA_ARGS=()
# Optional hard step cap (handy for smoke tests: MAX_STEPS=2 sbatch ...).
[ -n "${MAX_STEPS:-}" ] && EXTRA_ARGS+=(trainer.max_steps="${MAX_STEPS}")

# --- Checkpointing / run identity -------------------------------------------
derive_model_slug() {
  local p="$1" name=""
  [[ "$p" == *"/snapshots/"* ]] && name="$(sed -n 's#.*/models--[^/]*--\([^/]*\)/snapshots/.*#\1#p' <<<"$p")"
  [ -z "$name" ] && name="$(basename "$p")"
  tr '[:upper:]' '[:lower:]' <<<"$name" | sed 's/[^a-z0-9]\+/_/g; s/^_//; s/_$//'
}
MODEL_SLUG="$(derive_model_slug "${MODEL_PATH}")"  
EXP_NAME="${EXP_NAME:-${MODEL_SLUG}_${EXP_TAG}}"
PROJECT_NAME=VERVE
CKPT_DIR="${EASYR1_DIR}/verl_exp/${EXP_NAME}"
DONE_FILE="${CKPT_DIR}/TRAIN_DONE"

# --- Logging ------------------------------------------------------------------
# Compute nodes are offline, so W&B is ALWAYS offline (no API key needed); sync
# later from a login node:  wandb sync ${EASYR1_DIR}/wandb/offline-run-*
LOGGER='["file"]'
if [ "${USE_WANDB:-0}" = "1" ]; then
  LOGGER='["file","wandb"]'
  export WANDB_PROJECT="${PROJECT_NAME}"
  export WANDB_NAME="${EXP_NAME}"
  export WANDB_RUN_ID="${EXP_NAME}_${SLURM_JOB_ID}"  # stable id => one run across resubmissions
  export WANDB_RESUME=allow
  export WANDB_MODE=offline
  export WANDB_DIR="${EASYR1_DIR}"
fi

# --- Auto-resume across walltime (RESUME=1 in the launcher) ------------------
AUTORESUME="${RESUME:-0}"
ATTEMPT="${ATTEMPT:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-}"
RESUME_BUFFER_SECONDS="${RESUME_BUFFER_SECONDS:-1200}"  # leave time to flush ckpt + cleanup

# --- Offline HF + runtime env -------------------------------------------------
# Force every library into offline mode so nothing attempts a download /
# phone-home mid-training on the internet-less compute nodes.
export HF_HOME="${HF_HOME:?set HF_HOME to the Hugging Face cache that holds the policy and judge models}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export RAY_USAGE_STATS_ENABLED=0
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch 2.10 renamed the env var; the old name may be ignored.
export PYTORCH_ALLOC_CONF=expandable_segments:True
# vLLM's torch.compile cache goes on NODE-LOCAL scratch, not the shared GPFS
# home. A corrupted/interrupted write on GPFS previously left an all-null cache
# file that crashed rollout init with "source code string cannot contain null
# bytes". Node-local scratch avoids the shared-FS write races entirely.
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "${VLLM_CACHE_ROOT}"
# Keep per-process thread pools sane on a 288-core node shared by ray + torch.
export OMP_NUM_THREADS=16
# Never write .pyc files into the shared-FS python/venv from compute jobs:
# freshly-written pycache inodes are exactly what the 2026-07 filesystem
# incident corrupted (undeletable stale inodes that then poison every import).
export PYTHONDONTWRITEBYTECODE=1

# --- Log volume control -------------------------------------------------------
# slurm .out/.err otherwise grow by ~10 MB/step, dominated by tqdm redraws and
# vLLM INFO chatter (non-tty output turns every bar refresh into a new line).
# TQDM_MININTERVAL only throttles intermediate redraws; tqdm still emits a
# create + complete line for every (nested) bar instance, and in a non-tty the
# position= cursor codes turn each into new lines. Over many steps x ppo_epochs
# x mini/micro-batches that alone is MBs of .err. TQDM_DISABLE=1 silences every
# tqdm bar globally (verl's Compute log probs / Update policy / Running step /
# Train mini-batches), since none of those calls pass disable=False.
export TQDM_DISABLE=1
export TQDM_MININTERVAL=3600    # (kept as a fallback if TQDM_DISABLE is unset)
export VLLM_LOGGING_LEVEL=WARNING  # silence vLLM INFO logs (rollout engines + judge)
# Judge access log is SAMPLED, not disabled: 1 of every N chat/completions
# lines is kept (prefixed with the running request count), which preserves a
# picture of judge traffic for future data design without flooding the .out.
JUDGE_ACCESS_LOG_EVERY="${JUDGE_ACCESS_LOG_EVERY:-100}"

# Ray's raylet opens many gRPC sockets; a low nofile limit crashes it under
# multi-GPU load. Re-applied inside every per-node srun step (fresh shells).
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# Fail FAST if a model is not in the local cache: nodes are offline, so a
# missing snapshot would otherwise crash vLLM / verl minutes into startup.
check_model_local() {  # check_model_local <hf-id-or-local-dir>
  local m="$1"
  [ -d "${m}" ] && return 0                       # local checkpoint dir
  local cache="${HF_HOME}/hub/models--${m//\//--}"
  if [ -z "$(ls -A "${cache}/snapshots" 2>/dev/null)" ]; then
    echo "[error] model '${m}' not found in the local HF cache (${cache})." >&2
    echo "[error] Download it first on a login node:  HF_HOME=${HF_HOME} hf download ${m}" >&2
    exit 1
  fi
}
check_model_local "${MODEL_PATH}"
check_model_local "${JUDGE_MODEL}"

source "${VENV}/bin/activate"
cd "${EASYR1_DIR}"

# ----------------------------------------------------------------------------
# Discover allocated nodes. The batch script itself runs on the FIRST node.
# Carve: LAST node = dedicated judge; the rest train (Ray head on the first).
# ----------------------------------------------------------------------------
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NNODES_TOTAL="${#NODES[@]}"
N_JUDGE_NODES="${N_JUDGE_NODES:-1}"
if [ "${NNODES_TOTAL}" -lt $((N_JUDGE_NODES + 1)) ]; then
  echo "[error] need at least $((N_JUDGE_NODES + 1)) nodes (>=1 training + ${N_JUDGE_NODES} judge); got ${NNODES_TOTAL}" >&2
  exit 1
fi
JUDGE_NODES=("${NODES[@]:$((NNODES_TOTAL - N_JUDGE_NODES))}")
TRAIN_NODES=("${NODES[@]:0:$((NNODES_TOTAL - N_JUDGE_NODES))}")
HEAD_NODE="${TRAIN_NODES[0]}"
WORKER_NODES=("${TRAIN_NODES[@]:1}")
NNODES="${#TRAIN_NODES[@]}"     # TRAINING nodes only (what verl sees)

# Prefer the InfiniBand/IPoIB interface (ib*) for the head address; fall back
# to the first address from `hostname -I` if no ib interface is exposed.
HEAD_IP="$(ip -o -4 addr show | awk '$2 ~ /^ib/ {print $4}' | cut -d/ -f1 | head -1)"
[ -z "${HEAD_IP}" ] && HEAD_IP="$(hostname -I | awk '{print $1}')"
echo "Allocated nodes : ${NODES[*]}"
echo "Head node       : ${HEAD_NODE} (${HEAD_IP})"
echo "Worker nodes    : ${WORKER_NODES[*]:-<none>}"
echo "Judge node(s)   : ${JUDGE_NODES[*]} (GPUs ${JUDGE_GPUS} each, dp=4)"
echo "Training        : ${NNODES} nodes x ${N_GPUS_PER_NODE} GPUs = $((NNODES * N_GPUS_PER_NODE)) GPUs"
echo "Run identity    : ${EXP_NAME}  (ckpt: ${CKPT_DIR})"

# Helper to run a command on a specific node, inside the venv + project dir.
run_on() {  # run_on <node> <command...>
  local node="$1"; shift
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" \
    bash -lc "ulimit -n \$(ulimit -Hn) 2>/dev/null || true; source ${VENV}/bin/activate; cd ${EASYR1_DIR}; $*"
}

# Same as run_on but inside the judge environment.
run_on_judge() {  # run_on_judge <node> <command...>
  local node="$1"; shift
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" \
    bash -lc "ulimit -n \$(ulimit -Hn) 2>/dev/null || true; source ${JUDGE_VENV}/bin/activate; cd ${EASYR1_DIR}; $*"
}

# Helper: a node's ib IPv4 (fallback: first `hostname -I` address).
node_ip() {  # node_ip <node>
  local ip
  ip="$(srun --overlap --nodes=1 --ntasks=1 -w "$1" \
    bash -lc "ip -o -4 addr show | awk '\$2 ~ /^ib/ {print \$4}' | cut -d/ -f1 | head -1")"
  [ -z "${ip}" ] && ip="$(srun --overlap --nodes=1 --ntasks=1 -w "$1" hostname -I | awk '{print $1}')"
  echo "${ip}"
}

# ----------------------------------------------------------------------------
# 0) Node preflight. The cluster's shared FS has intermittently served stale
# handles for recently-written dirs (NHC drains nodes for check_stale_file all
# day 2026-07-23); on affected not-yet-drained nodes even `import http.client`
# fails inside the venv, killing ray/vllm minutes into startup. Import-test the
# venv on EVERY allocated node first (~15 s): if any node is sick, exit fast
# and (when RESUME=1) resubmit ourselves excluding the accumulated bad nodes.
# EXCLUDE_NODES rides along via sbatch --export=ALL across resubmissions.
# ----------------------------------------------------------------------------
declare -a PREFLIGHT_BAD=()
declare -a PREFLIGHT_PIDS=()
PREFLIGHT_TMP="$(mktemp -d)"
for n in "${NODES[@]}"; do
  ( srun --overlap --nodes=1 --ntasks=1 -w "${n}" \
      bash -lc "source ${VENV}/bin/activate && python -c 'import http.client, urllib3, requests, ray'" \
      >/dev/null 2>&1 || touch "${PREFLIGHT_TMP}/${n}" ) &
  PREFLIGHT_PIDS+=("$!")
done
for pid in "${PREFLIGHT_PIDS[@]}"; do wait "${pid}" || true; done
for n in "${NODES[@]}"; do
  [ -e "${PREFLIGHT_TMP}/${n}" ] && PREFLIGHT_BAD+=("${n}")
done
rm -rf "${PREFLIGHT_TMP}"
if [ "${#PREFLIGHT_BAD[@]}" -gt 0 ]; then
  bad_csv="$(IFS=,; echo "${PREFLIGHT_BAD[*]}")"
  export EXCLUDE_NODES="${EXCLUDE_NODES:+${EXCLUDE_NODES},}${bad_csv}"
  echo "[preflight] SICK NODES (venv import failed): ${bad_csv}" >&2
  echo "[preflight] cumulative exclude list: ${EXCLUDE_NODES}" >&2
  if [ "${AUTORESUME}" = "1" ] && [ -n "${RESUBMIT_SCRIPT}" ] && [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "[preflight] resubmitting attempt $((ATTEMPT + 1))/${MAX_ATTEMPTS} with --exclude=${EXCLUDE_NODES}" >&2
    sbatch --exclude="${EXCLUDE_NODES}" \
      --export=ALL,RESUME=1,ATTEMPT=$((ATTEMPT + 1)),MAX_ATTEMPTS="${MAX_ATTEMPTS}",EXCLUDE_NODES="${EXCLUDE_NODES}" \
      "${RESUBMIT_SCRIPT}"
  else
    echo "[preflight] not resubmitting (RESUME!=1, no RESUBMIT_SCRIPT, or MAX_ATTEMPTS reached)." >&2
  fi
  exit 1
fi
echo "[preflight] all ${NNODES_TOTAL} nodes healthy (venv imports OK)"

# Teardown: stop ray everywhere and kill background steps on exit.
cleanup() {
  echo "[cleanup] tearing down ray + judge server"
  ray stop --force >/dev/null 2>&1 || true
  # Every srun here talks to slurmctld; during the 2026-08-21 controller outage
  # these hung forever and the crashed job idled its 7-node allocation for 11.5h
  # until walltime. Bound them, and end with a self-scancel as a backstop.
  for wn in "${WORKER_NODES[@]:-}"; do
    [ -n "${wn}" ] && timeout 60 srun --overlap --nodes=1 --ntasks=1 -w "${wn}" \
      bash -lc "source ${VENV}/bin/activate; ray stop --force" >/dev/null 2>&1 || true
  done
  for pid in "${BG_PIDS[@]:-}"; do kill "${pid}" >/dev/null 2>&1 || true; done
  [ -n "${SLURM_JOB_ID:-}" ] && timeout 60 scancel "${SLURM_JOB_ID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
declare -a BG_PIDS=()

# ----------------------------------------------------------------------------
# 1) Launch judge vLLM server(s) per judge node. Backgrounded srun steps.
#    The reward function load-balances across all endpoints in JUDGE_API_BASES.
#
#    JUDGE_SHARDS_PER_NODE=1 (default): one server on all 4 GPUs with
#      JUDGE_PARALLEL_ARGS (default --data-parallel-size 4). Fine for DENSE
#      judges (Qwen3-8B).
#    JUDGE_SHARDS_PER_NODE=4: four independent single-GPU TP=1 servers on
#      ports JUDGE_PORT..JUDGE_PORT+3 — REQUIRED for MoE judges
#      (Qwen3-30B-A3B): both multi-GPU modes (dp=4, tp=4) crashed on it in
#      our setup, TP=1 has no inter-GPU path at all. Each shard gets its own
#      VLLM_PORT range: servers starting together can otherwise pick the same
#      internal port, and the one that loses dies at startup.
# ----------------------------------------------------------------------------
JUDGE_PARALLEL_ARGS="${JUDGE_PARALLEL_ARGS:---data-parallel-size 4}"
JUDGE_SHARDS_PER_NODE="${JUDGE_SHARDS_PER_NODE:-1}"
JUDGE_AWK_SAMPLER="awk -v every=${JUDGE_ACCESS_LOG_EVERY} \
  '{ if (\$0 ~ \"POST /v1/chat/completions\") { n++; if (n % every != 1) next; \$0 = \"[judge-access #\" n \"] \" \$0 } print; fflush(); }'"
declare -a JUDGE_URLS=()
for jn in "${JUDGE_NODES[@]}"; do
  jip="$(node_ip "${jn}")"
  if [ "${JUDGE_SHARDS_PER_NODE}" -gt 1 ]; then
    for g in $(seq 0 $((JUDGE_SHARDS_PER_NODE - 1))); do
      port=$((JUDGE_PORT + g))
      echo "[judge] starting shard on ${jn} (GPU ${g}, TP=1, port ${port})"
      run_on_judge "${jn}" \
        "CUDA_VISIBLE_DEVICES=${g} VLLM_PORT=$((20000 + 100 * g)) vllm serve ${JUDGE_MODEL} \
           --port ${port} --gpu-memory-utilization 0.9 --max-model-len 32768 \
           --served-model-name judge \
           --trust-remote-code \
           2>&1 | ${JUDGE_AWK_SAMPLER}" &
      BG_PIDS+=("$!")
      JUDGE_URLS+=("http://${jip}:${port}")
    done
  else
    echo "[judge] starting on dedicated node ${jn} (GPUs ${JUDGE_GPUS}, ${JUDGE_PARALLEL_ARGS})"
    run_on_judge "${jn}" \
      "CUDA_VISIBLE_DEVICES=${JUDGE_GPUS} vllm serve ${JUDGE_MODEL} \
         --port ${JUDGE_PORT} --gpu-memory-utilization 0.9 --max-model-len 32768 \
         ${JUDGE_PARALLEL_ARGS} --served-model-name judge \
         --trust-remote-code \
         2>&1 | ${JUDGE_AWK_SAMPLER}" &
    BG_PIDS+=("$!")
    JUDGE_URLS+=("http://${jip}:${JUDGE_PORT}")
  fi
done

# Export JUDGE_API_BASES NOW, *before* `ray start`, so every Ray worker
# process inherits it (the reward function reads it at import time).
JUDGE_API_BASES="$(printf '%s/v1,' "${JUDGE_URLS[@]}")"
export JUDGE_API_BASES="${JUDGE_API_BASES%,}"
echo "[judge] JUDGE_API_BASES=${JUDGE_API_BASES}"

# ----------------------------------------------------------------------------
# 2) Bootstrap the Ray cluster on the TRAINING nodes (all 4 GPUs each).
#    Head runs locally (persists with this script); workers via backgrounded
#    srun steps kept alive with `--block`. The judge node does NOT join Ray.
# ----------------------------------------------------------------------------
# RAY_ADDRESS is exported further down for the driver, and the walltime
# resubmit carries the whole environment with `sbatch --export=ALL`. A
# continuation therefore inherits the PREVIOUS job's head IP, and every
# `ray status` in the barrier below queries a dead GCS instead of the cluster
# this job just built: 120 retries x a ~40s GCS timeout ~= 90 min, then
# "[wait] ERROR: ray pool did not reach N GPUs" and rc=1 (jobs 1497747/1497749,
# 2026-08-26 -- the first clean walltime boundary this setup ever reached, so
# it had never fired before). Clear it; it is re-derived below.
unset RAY_ADDRESS
echo "[ray] starting head on ${HEAD_NODE}:${RAY_PORT} (GPUs ${TRAIN_GPUS})"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" ray start --head \
  --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
  --num-gpus="${N_GPUS_PER_NODE}"

for wn in "${WORKER_NODES[@]:-}"; do
  [ -z "${wn}" ] && continue
  echo "[ray] joining worker ${wn} -> ${HEAD_IP}:${RAY_PORT} (GPUs ${TRAIN_GPUS})"
  run_on "${wn}" \
    "CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} ray start --address=${HEAD_IP}:${RAY_PORT} \
       --num-gpus=${N_GPUS_PER_NODE} --block" &
  BG_PIDS+=("$!")
done

# ----------------------------------------------------------------------------
# 3) Barrier: wait until the Ray pool has all training GPUs AND the judge
#    endpoint answers, each with a timeout. (The judge takes minutes to load.)
# ----------------------------------------------------------------------------
EXPECTED_GPUS=$((NNODES * N_GPUS_PER_NODE))
echo "[wait] expecting ${EXPECTED_GPUS} GPUs in the Ray pool"
for i in $(seq 1 120); do
  # `|| true` — under `set -euo pipefail` a transient `ray status` failure
  # (GCS still assembling) or sed's SIGPIPE from `head -1` silently killed the
  # whole script here (2026-08-15, jobs 1381110/1381356). A failed poll must
  # simply mean "retry in 5s", which is this loop's entire purpose.
  total="$( { ray status 2>/dev/null || true; } | { sed -n 's#.*/\([0-9.]*\) GPU#\1#p' || true; } | head -1 )" || true
  total="${total%.*}"
  if [ -n "${total}" ] && [ "${total}" -ge "${EXPECTED_GPUS}" ]; then
    echo "[wait] ray pool ready: ${total} GPUs"; break
  fi
  [ "${i}" -eq 120 ] && { echo "[wait] ERROR: ray pool did not reach ${EXPECTED_GPUS} GPUs"; ray status || true; exit 1; }
  sleep 5
done

for u in "${JUDGE_URLS[@]}"; do
  echo "[wait] judge ${u} ..."
  for i in $(seq 1 360); do
    if curl -sf "${u}/v1/models" >/dev/null 2>&1; then
      echo "[wait] judge ${u} ready"; break
    fi
    [ "${i}" -eq 360 ] && { echo "[wait] ERROR: judge ${u} not ready"; exit 1; }
    sleep 5
  done
done

# ----------------------------------------------------------------------------
# 4) Run the training driver on the head, attached to the running Ray cluster.
#    Batch sizes MUST divide evenly across the training world size
#    (n_gpus_per_node * nnodes = 12 GPUs for the 4-node launcher).
# ----------------------------------------------------------------------------
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

# If a previous attempt already finished, nothing to do (resumed job no-op).
if [ "${AUTORESUME}" = "1" ] && [ -f "${DONE_FILE}" ]; then
  echo "[done] ${DONE_FILE} exists -> training already complete; exiting."
  exit 0
fi

# Wrap the driver in `timeout` so we regain control ~RESUME_BUFFER_SECONDS
# before walltime and can resubmit to continue from the last checkpoint.
TIMEOUT_PREFIX=()
if [ "${AUTORESUME}" = "1" ] && [ -n "${SLURM_JOB_END_TIME:-}" ]; then
  now="$(date +%s)"
  budget=$(( SLURM_JOB_END_TIME - now - RESUME_BUFFER_SECONDS ))
  [ "${budget}" -lt 600 ] && budget=600
  echo "[resume] AUTORESUME on (attempt ${ATTEMPT}/${MAX_ATTEMPTS}); driver time budget=${budget}s"
  TIMEOUT_PREFIX=(timeout --signal=TERM "${budget}s")
fi

echo "[train] launching verl.trainer.main (RAY_ADDRESS=${RAY_ADDRESS}, logger=${LOGGER})"
echo "[train] batch sizes: rollout=${ROLLOUT_BS} global=${GLOBAL_BS} mini=${MINI_BS} epochs=${TOTAL_EPOCHS}"

set +e
"${TIMEOUT_PREFIX[@]}" python3 -m verl.trainer.main \
    config=./examples/perception_config.yaml \
    data.train_files="${TRAIN_PARQUET}" \
    data.image_dir="${IMAGE_DIR}" \
    data.mini_rollout_batch_size="${MINI_BS}" \
    data.max_prompt_length=25000 \
    data.max_response_length=2048 \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.rollout.max_num_batched_tokens=28000 \
    worker.rollout.disable_tqdm=true \
    worker.reward.reward_function="${REWARD_FN:-./examples/reward_function/perception_multinode.py:compute_score}" \
    algorithm.disable_kl=True \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_checkpoint_path="${CKPT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_limit="${SAVE_LIMIT}" \
    trainer.find_last_checkpoint=true \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "${EXTRA_ARGS[@]}"
rc=$?
set -e

# ----------------------------------------------------------------------------
# Completion / auto-resume decision.
#   rc == 0            -> training finished all steps -> mark DONE.
#   rc in {124,143}    -> hit the timeout (walltime) -> resubmit to continue.
#   other              -> real crash -> do NOT resubmit (avoid loops).
# ----------------------------------------------------------------------------
if [ "${rc}" -eq 0 ]; then
  mkdir -p "${CKPT_DIR}"; touch "${DONE_FILE}"
  echo "[done] training complete (rc=0); wrote ${DONE_FILE}"
elif [ "${AUTORESUME}" = "1" ] && { [ "${rc}" -eq 124 ] || [ "${rc}" -eq 143 ]; }; then
  echo "[resume] driver stopped at walltime budget (rc=${rc})."
  if [ -n "${RESUBMIT_SCRIPT}" ] && [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "[resume] submitting continuation attempt $((ATTEMPT + 1))/${MAX_ATTEMPTS}"
    sbatch ${EXCLUDE_NODES:+--exclude=${EXCLUDE_NODES}} \
      --export=ALL,RESUME=1,ATTEMPT=$((ATTEMPT + 1)),MAX_ATTEMPTS="${MAX_ATTEMPTS}"${EXCLUDE_NODES:+,EXCLUDE_NODES=${EXCLUDE_NODES}} \
      "${RESUBMIT_SCRIPT}"
  else
    echo "[resume] not resubmitting (no RESUBMIT_SCRIPT or reached MAX_ATTEMPTS=${MAX_ATTEMPTS})."
  fi
else
  echo "[error] driver exited rc=${rc} (not a timeout); not resubmitting."
  exit "${rc}"
fi
