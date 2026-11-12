#!/bin/bash
# Two-node offload comparison on Pinnacles (test partition: MaxNodes=2, 2x L40S per node -> 4 GPUs, SP=4).
# usage: sbatch sb_mn.sh <model> <cfg> <arms> [KEY=VALUE ...]      e.g. sbatch sb_mn.sh wan 41 no,old NSYS=0
# Each arm is one job step on both nodes (srun --wait kills the worker-only node 90 s after node 0 exits).
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --exclude=gnode017
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
m="$1"; c="$2"; arms="$3"; shift 3
for kv in "$@"; do case "$kv" in *=*) export "$kv" ;; esac; done
WS=/data/hanmeng/chunkflow
export WS NNODES="$SLURM_NNODES" GPUS_PER_NODE="${GPUS_PER_NODE:-2}" NSYS="${NSYS:-0}" STEPS="${STEPS:-10}" CHUNK_MB="${CHUNK_MB:-4}" TAG="${TAG:-}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
# Pinnacles: IPoIB addresses are not routable for the NCCL/gloo bootstrap -> use the Ethernet interface; data over the IB HCA
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-en0}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-en0}" NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
TOTAL=$(( NNODES * GPUS_PER_NODE )); RUN="${m}_c${c}_mn${TOTAL}${TAG:+_$TAG}"; export OUT="$WS/results/$RUN"; mkdir -p "$OUT/logs"
echo "== sb_mn: $RUN arms=$arms nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ') master=$MASTER_ADDR total_gpus=$TOTAL ib_disable=${NCCL_IB_DISABLE:-0} $(date) =="
srun -N "$NNODES" --ntasks-per-node=1 bash -c 'echo "$(hostname): $(nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader | tr "\n" ";") ib: $(ls /sys/class/infiniband 2>/dev/null | tr "\n" " ")"' | tee "$OUT/logs/nodes.txt"
# verbs-level RDMA check between the two nodes (ib_write_bw server on node 0, client on node 1), 20 s cap, informational
if [[ "$NNODES" -gt 1 && "${IBCHECK:-1}" == 1 ]]; then
    n0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p); n1=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 2p)
    ( srun -N1 -n1 -w "$n0" --overlap bash -c "command -v ib_write_bw >/dev/null && timeout 25 ib_write_bw -d mlx5_0 -s 1048576 -n 200 --report_gbits 2>&1 | tail -3 || echo 'ib_write_bw not installed on $n0'" &
      sleep 4; srun -N1 -n1 -w "$n1" --overlap bash -c "command -v ib_write_bw >/dev/null && timeout 20 ib_write_bw -d mlx5_0 -s 1048576 -n 200 --report_gbits $n0 2>&1 | tail -3 || echo 'ib_write_bw not installed on $n1'"; wait ) 2>&1 | tee "$OUT/logs/ib_write_bw.txt"
fi
# cross-node NCCL bandwidth sanity (all_to_all / send-recv per rank), informational, multi-node only; raw log kept
if [[ "${NCCLBW:-1}" == 1 && "$NNODES" -gt 1 ]]; then
    srun -N "$NNODES" --ntasks-per-node=1 --wait=60 bash -c "source $WS/sglang/.venv/bin/activate && NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET torchrun --nnodes $NNODES --node_rank \$SLURM_NODEID --master_addr $MASTER_ADDR --master_port 29399 --nproc_per_node $GPUS_PER_NODE $WS/nccl_bw.py" > "$OUT/logs/nccl_bw_raw.txt" 2>&1
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
