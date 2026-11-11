#!/bin/bash
# Runbook §3 loop for one platform. PLAT=l40s8 (2 nodes x 4, run on node 0 of the allocation, nested srun per arm)
# or PLAT=h100 (1 node x 2). Resumable after preemption: an arm whose rank-0 log already has a stage average is skipped.
source /scratch/yx2432/rebuttal_mn/site.sh
for i in $(seq 1 240); do [ -e $WS/ENV_READY ] && break; sleep 15; done          # wait (<=1 h) for the venv install
[ -e $WS/ENV_READY ] || { echo "env not ready, giving up"; exit 3; }
PLAT=${PLAT:?}; RUNLOG=$WS/results/driver_$PLAT.txt
log() { echo "[$(date '+%F %T')] $*" | tee -a $RUNLOG; }
( cd $WS/sglang && git pull --ff-only 2>&1 | tail -1; echo "commit $(git log --oneline -1)" ) | tee -a $RUNLOG
# GPU-side sanity (cannot be checked on the login node: no driver there)
source $WS/sglang/.venv/bin/activate
python -c "import torch, sgl_kernel; print('ENV', torch.__version__, torch.cuda.get_device_name(0), 'sgl_kernel ok')" 2>&1 | tail -1 | tee -a $RUNLOG
timeout 300 sglang generate --help 2>&1 | grep -oE -- "--(nnodes|node-rank|dist-init-addr)" | sort -u | tr '\n' ' ' | sed 's/^/flags: /' | tee -a $RUNLOG; echo | tee -a $RUNLOG
RUNMN=$WS/sglang/offload_profiling/pinnacles_mech/run_mn_arm.sh
if [ $PLAT = l40s8 ]; then
  export NNODES=2 GPUS_PER_NODE=4 TAG=l40s8 MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
  LAUNCH=(srun --overlap --jobid=$SLURM_JOB_ID -N 2 -n 2 --ntasks-per-node=1 --wait=90 --kill-on-bad-exit=0)
  TOPO=(srun --overlap --jobid=$SLURM_JOB_ID -N 2 -n 2 --ntasks-per-node=1 bash -c 'echo "== $(hostname -s)"; nvidia-smi topo -m')
else
  export NNODES=1 GPUS_PER_NODE=2 TAG=h100 MASTER_ADDR=$(hostname -s)
  LAUNCH=(); TOPO=(bash -c 'echo "== $(hostname -s)"; nvidia-smi topo -m')
fi
log "platform $PLAT job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR"
[ -s $WS/results/topo_$PLAT.txt ] || "${TOPO[@]}" > $WS/results/topo_$PLAT.txt 2>&1
first=1
export CHUNK_MB=16    # run_mn_arm.sh defaults to 4; the paper and all comparisons use 16 MB chunks
if [ $PLAT = h100 ]; then ARMS="no old stock gate"; else ARMS="no old"; fi   # stock/gate only on the H100 platform
for f in 41 81 121 161; do for arm in $ARMS; do
  OUTD=$WS/results/wan_c${f}_mn$((NNODES*GPUS_PER_NODE))_${TAG}
  if grep -q "average time per step" $OUTD/logs/$arm.log 2>/dev/null; then log "skip $f $arm (already has a result)"; continue; fi
  if [ $first = 1 ]; then export NCCL_DEBUG=INFO; else export NCCL_DEBUG=WARN; fi   # one INFO arm per start: records the transport line
  log "run $f $arm NCCL_DEBUG=$NCCL_DEBUG"
  "${LAUNCH[@]}" bash $RUNMN wan $f $arm 7
  log "end $f $arm rc=$? $(grep -oE 'average time per step: [0-9.]+' $OUTD/logs/$arm.log 2>/dev/null | tail -1)"
  if [ $PLAT = l40s8 ]; then srun --overlap --jobid=$SLURM_JOB_ID -N 2 -n 2 --ntasks-per-node=1 bash -c 'pgrep -u $USER -f "[b]in/sglang generate" >/dev/null && { pkill -u $USER -f "[b]in/sglang generate"; sleep 5; pkill -9 -u $USER -f "[b]in/sglang generate"; echo "killed leftover on $(hostname -s)"; }; true' | tee -a $RUNLOG; fi
  first=0
done; done
log "platform $PLAT finished"
