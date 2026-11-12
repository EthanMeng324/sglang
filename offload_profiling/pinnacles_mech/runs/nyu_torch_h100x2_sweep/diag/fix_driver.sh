#!/bin/bash
# Verify the gate fix (deferred buffer release). run_mn_arm_diag.sh (--seed 42, saves video), CHUNK_MB=16, TAG=h100fix.
source /scratch/yx2432/rebuttal_mn/site.sh; source $WS/sglang/.venv/bin/activate
export NNODES=1 GPUS_PER_NODE=2 MASTER_ADDR=$(hostname -s) NCCL_DEBUG=WARN CHUNK_MB=16
G=$WS/diag; L=$G/fix_log.txt; RUN=$WS/run_mn_arm_diag.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a $L; }
log "fix check start job=$SLURM_JOB_ID host=$(hostname -s) commit=$(cd $WS/sglang && git log --oneline -1 | cut -c1-60)"
log "IMMEDIATE_RELEASE: $(grep -n IMMEDIATE_RELEASE $WS/sglang/python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py | head -3 | tr '\n' ' ')"
arm() { local f=$1 a=$2 tag=$3; local out=$WS/results/wan_c${f}_mn2_${tag}
  if grep -q "average time per step" $out/logs/$a.log 2>/dev/null; then log "skip $f $a $tag (has result)"; return; fi
  mkdir -p $out/logs; nvidia-smi dmon -s pc -d 1 -o T > $out/logs/dmon_$a.txt 2>&1 & local dm=$!
  log "run f=$f arm=$a TAG=$tag"; env TAG=$tag bash $RUN wan $f $a 7; local rc=$?
  kill $dm 2>/dev/null; wait $dm 2>/dev/null
  log "end f=$f arm=$a TAG=$tag rc=$rc $(grep -oE 'average time per step: [0-9.]+' $out/logs/$a.log | tail -1)"; }
arm 41 no h100fix; arm 41 gate h100fix; arm 161 gate h100fix; arm 161 stock h100fix
R=$WS/results
{ echo "== PSNR vs no (diag/psnr.py)"
  echo "gate41_fix vs no41_fix: $(python $G/psnr.py $R/wan_c41_mn2_h100fix/gate.out.mp4 $R/wan_c41_mn2_h100fix/no.out.mp4 2>&1 | tail -1)"
  echo "gate161_fix vs no161_d1: $(python $G/psnr.py $R/wan_c161_mn2_h100fix/gate.out.mp4 $R/wan_c161_mn2_h100d1/no.out.mp4 2>&1 | tail -1)"
  echo "stock161_fix vs no161_d1: $(python $G/psnr.py $R/wan_c161_mn2_h100fix/stock.out.mp4 $R/wan_c161_mn2_h100d1/no.out.mp4 2>&1 | tail -1)"
  echo "no161_d1 vs no161_d2 (baseline): $(python $G/psnr.py $R/wan_c161_mn2_h100d1/no.out.mp4 $R/wan_c161_mn2_h100d2/no.out.mp4 2>&1 | tail -1)"; } | tee -a $L
log "fix check finished"
