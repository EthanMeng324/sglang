#!/usr/bin/env bash
# One arm of the offload comparison on ONE node of a multi-node run. Run the same command on every
# node at the same time (srun, pdsh, or one ssh per node); node 0 drives the request, the others only
# host workers.
#
# usage: bash run_mn_arm.sh <wan|hunyuan|flux> <cfg> <arm> [port_offset]
#   arms: no | old | stock | launch | gpu | gate   (old = paper "Layerwise" baseline, stock = upstream)
# required env: MASTER_ADDR (hostname/IP of node 0, reachable from all nodes)
# env with defaults: NNODES=2 GPUS_PER_NODE=4 NODE_RANK=${SLURM_NODEID:-0} STEPS=10 NSYS=0 CHUNK_MB=4
#                    OUT=$WS/results/<model>_c<cfg>_mn<TAG>  WS=/data/hanmeng/chunkflow  ATTENTION_BACKEND=fa
#                    MASTER_PORT=29500 (+port_offset)  ARM_TIMEOUT=1800  DIST_TIMEOUT=900
set -uo pipefail
MODEL="${1:?model}"; CFG="${2:?cfg}"; ARM="${3:?arm}"; OFF="${4:-0}"
WS="${WS:-/data/hanmeng/chunkflow}"
NNODES="${NNODES:-2}"; GPUS_PER_NODE="${GPUS_PER_NODE:-4}"; NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
TOTAL=$(( NNODES * GPUS_PER_NODE ))
STEPS="${STEPS:-10}"; NSYS="${NSYS:-0}"; CHUNK_MB="${CHUNK_MB:-4}"; ATTN="${ATTENTION_BACKEND:-fa}"
TAG="${TAG:-}"; RUN="${MODEL}_c${CFG}_mn${TOTAL}${TAG:+_$TAG}"; OUT="${OUT:-$WS/results/$RUN}"
PORT=$(( ${MASTER_PORT:-29500} + OFF ))
: "${MASTER_ADDR:?MASTER_ADDR (node 0) is required}"
mkdir -p "$OUT/logs" "$OUT/profiles"

source "$WS/sglang/.venv/bin/activate"
export HF_HOME="${HF_HOME:-$WS/hf}" PYTHONUNBUFFERED=1
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WS/.cache/torchinductor}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WS/.cache/triton}"
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1 SGLANG_DIFFUSION_STAGE_LOGGING=1 SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES=1 SGLANG_WAN_MOCK_COMM_ENABLE=0
# cross-node NCCL/gloo: set NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME / NCCL_IB_HCA in the environment if the
# default interface selection is wrong for this cluster (see the runbook)
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "$NNODES" -gt 1 ]]; then
    # Pinnacles: the IPoIB (ib0) addresses are not routable between nodes, so the NCCL/gloo bootstrap
    # must use the Ethernet interface; the data path still uses the IB HCA (verbs).
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-en0}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-en0}" NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
fi

PROMPT="A cat walks on the grass, realistic"
case "$MODEL" in
    wan)     MP="$WS/models/Wan2.2-TI2V-5B-Diffusers"; MFLAGS=(--num-frames "$CFG" --height 704 --width 1280 --guidance-scale 4.0 --guidance-scale-2 3.0) ;;
    hunyuan) MP="$WS/models/HunyuanVideo"; MFLAGS=(--num-frames "$CFG" --height 720 --width 1280 --guidance-scale 1.0) ;;
    flux)    MP="$WS/models/FLUX.1-dev"; MFLAGS=(--height 1024 --width 1024 --num-outputs-per-prompt "$CFG") ;;
    *) echo "unknown model $MODEL"; exit 2 ;;
esac

offload=(--dit-layerwise-offload true --dit-offload-prefetch-size 1)
cf=(SGLANG_DIT_COMM_AWARE_OFFLOAD=1 SGLANG_DIT_PHASE_AWARE_PREFETCH=0 SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB="$CHUNK_MB" SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST=0)
env_vars=(); flags=()
case "$ARM" in
    no)     env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0); flags+=(--dit-layerwise-offload false) ;;
    old)    env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0 SGLANG_DIT_OFFLOAD_COMM_QUIESCE=1); flags+=("${offload[@]}") ;;
    stock)  env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0 SGLANG_DIT_OFFLOAD_COMM_QUIESCE=0); flags+=("${offload[@]}") ;;
    launch|kernel|none|gpu|auto|gate)
            env_vars+=("${cf[@]}" SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE="$ARM" SGLANG_DIT_OFFLOAD_RESIDENT_RATIO=); flags+=("${offload[@]}") ;;
    *) echo "unknown arm $ARM"; exit 2 ;;
esac

cmd=(sglang generate
    --model-path "$MP" --text-encoder-cpu-offload --pin-cpu-memory
    --num-gpus "$TOTAL" --ulysses-degree "$TOTAL" --attention-backend "$ATTN" --enable-torch-compile
    --nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$MASTER_ADDR:$PORT" --dist-timeout "${DIST_TIMEOUT:-900}"
    --prompt "$PROMPT" --num-inference-steps "$STEPS" "${MFLAGS[@]}"
    --port "$((30000 + OFF))" --scheduler-port "$((5700 + OFF))"
    --output-path "$OUT" --output-file-name "${ARM}.out" --no-save-output
    "${flags[@]}")
suffix=""; [[ "$NODE_RANK" != 0 ]] && suffix="_node${NODE_RANK}"
[[ "$NODE_RANK" == 0 ]] && cmd+=(--perf-dump-path "$OUT/profiles/perf_${ARM}.json")
if [[ "$NSYS" == 1 ]]; then
    cmd=(nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --trace-fork-before-exec=true
         --kill=none --force-overwrite=true --output="$OUT/profiles/${ARM}${suffix}_nsys" "${cmd[@]}")
fi
log="$OUT/logs/${ARM}${suffix}.log"
echo "[$(date +%H:%M:%S)] node=$NODE_RANK/$NNODES host=$(hostname) ARM=$ARM total_gpus=$TOTAL master=$MASTER_ADDR:$PORT env: ${env_vars[*]}" | tee -a "$OUT/logs/run.txt"
t0=$(date +%s); rc=0
timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT:-1800}" env "${env_vars[@]}" "${cmd[@]}" > "$log" 2>&1 || rc=$?
avg=$(grep -oE "average time per step: [0-9.]+" "$log" | tail -1)
echo "[$(date +%H:%M:%S)] node=$NODE_RANK ARM=$ARM rc=$rc wall=$(( $(date +%s) - t0 ))s ${avg:-no-avg}" | tee -a "$OUT/logs/run.txt"
(( rc != 0 )) && grep -m3 -iE "out of memory|Traceback|Error" "$log" | head -3
exit "$rc"
