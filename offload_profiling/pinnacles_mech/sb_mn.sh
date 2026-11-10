#!/bin/bash
# Two-node offload comparison on Pinnacles (test partition: MaxNodes=2, 2x L40S per node -> 4 GPUs, SP=4).
# usage: sbatch sb_mn.sh <model> <cfg> <arms> [KEY=VALUE ...]      e.g. sbatch sb_mn.sh wan 41 no,old NSYS=0
# Each arm is one job step on both nodes (srun --wait kills the worker-only node 90 s after node 0 exits).
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
m="$1"; c="$2"; arms="$3"; shift 3
for kv in "$@"; do case "$kv" in *=*) export "$kv" ;; esac; done
WS=/data/hanmeng/chunkflow
export WS NNODES="$SLURM_NNODES" GPUS_PER_NODE="${GPUS_PER_NODE:-2}" NSYS="${NSYS:-0}" STEPS="${STEPS:-10}" CHUNK_MB="${CHUNK_MB:-4}" TAG="${TAG:-}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
TOTAL=$(( NNODES * GPUS_PER_NODE )); RUN="${m}_c${c}_mn${TOTAL}${TAG:+_$TAG}"; export OUT="$WS/results/$RUN"; mkdir -p "$OUT/logs"
echo "== sb_mn: $RUN arms=$arms nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ') master=$MASTER_ADDR total_gpus=$TOTAL $(date) =="
srun -N "$NNODES" --ntasks-per-node=1 bash -c 'echo "$(hostname): $(nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader | tr "\n" ";") ib: $(ls /sys/class/infiniband 2>/dev/null | tr "\n" " ")"' | tee "$OUT/logs/nodes.txt"
# cross-node NCCL bandwidth sanity (all_to_all / send-recv per rank), informational
if [[ "${NCCLBW:-1}" == 1 ]]; then
    srun -N "$NNODES" --ntasks-per-node=1 --wait=60 bash -c "source $WS/sglang/.venv/bin/activate && NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-en0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-en0} NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0} torchrun --nnodes $NNODES --node_rank \$SLURM_NODEID --master_addr $MASTER_ADDR --master_port 29399 --nproc_per_node $GPUS_PER_NODE $WS/nccl_bw.py" > "$OUT/logs/nccl_bw_raw.txt" 2>&1
    grep -E "===|GB/s|via NET|NET/|Error|error" "$OUT/logs/nccl_bw_raw.txt" | sed -E 's/.*NCCL INFO //' | sort | uniq -c | sort -rn | head -40 | tee "$OUT/logs/nccl_bw.txt"
fi
off=0
IFS=',' read -r -a LIST <<<"$arms"
for arm in "${LIST[@]}"; do
    off=$(( off + 7 ))
    srun -N "$NNODES" --ntasks-per-node=1 --wait=90 --kill-on-bad-exit=0 bash "$WS/run_mn_arm.sh" "$m" "$c" "$arm" "$off"
    sleep 5
done
echo "== results ($OUT/logs/run.txt)"; cat "$OUT/logs/run.txt"
echo "== DONE sb_mn $RUN $(date)"
