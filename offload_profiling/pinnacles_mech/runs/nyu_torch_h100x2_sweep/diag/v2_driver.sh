#!/bin/bash
# Verify the copy-stream allocation fix (67e62bdff): gate at 41/81/121/161, TAG=h100v2, run_mn_arm_diag.sh
# (--seed 42, saves video), CHUNK_MB=16. no is run only where no saved no-video exists (81, 121).
source /scratch/yx2432/rebuttal_mn/site.sh; source $WS/sglang/.venv/bin/activate
export NNODES=1 GPUS_PER_NODE=2 MASTER_ADDR=$(hostname -s) NCCL_DEBUG=WARN CHUNK_MB=16
G=$WS/diag; L=$G/v2_log.txt; RUN=$WS/run_mn_arm_diag.sh; R=$WS/results
log() { echo "[$(date '+%F %T')] $*" | tee -a $L; }
log "v2 start job=$SLURM_JOB_ID host=$(hostname -s) commit=$(cd $WS/sglang && git log --oneline -1 | cut -c1-70)"
log "fix line: $(grep -n 'allocated on the copy stream' $WS/sglang/python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py | head -1)"
arm() { local f=$1 a=$2 tag=$3; local out=$R/wan_c${f}_mn2_${tag}
  if grep -q "average time per step" $out/logs/$a.log 2>/dev/null; then log "skip $f $a $tag (has result)"; return; fi
  mkdir -p $out/logs; nvidia-smi dmon -s pc -d 1 -o T > $out/logs/dmon_$a.txt 2>&1 & local dm=$!
  log "run f=$f arm=$a TAG=$tag"; env TAG=$tag bash $RUN wan $f $a 7; local rc=$?
  kill $dm 2>/dev/null; wait $dm 2>/dev/null
  log "end f=$f arm=$a TAG=$tag rc=$rc $(grep -oE 'average time per step: [0-9.]+' $out/logs/$a.log | tail -1)"; }
for f in 41 81 121 161; do
  [ $f = 81 ] || [ $f = 121 ] && arm $f no h100v2
  arm $f gate h100v2
done
NOV() { case $1 in 41) echo $R/wan_c41_mn2_h100fix/no.out.mp4;; 161) echo $R/wan_c161_mn2_h100d1/no.out.mp4;; *) echo $R/wan_c$1_mn2_h100v2/no.out.mp4;; esac; }
{ echo "== PSNR gate(v2) vs no (diag/psnr.py)"
  for f in 41 81 121 161; do echo "f$f gate_v2 vs $(NOV $f | sed "s#$R/##"): $(python $G/psnr.py $R/wan_c${f}_mn2_h100v2/gate.out.mp4 $(NOV $f) 2>&1 | tail -1)"; done
  echo "baseline no161_d1 vs no161_d2: $(python $G/psnr.py $R/wan_c161_mn2_h100d1/no.out.mp4 $R/wan_c161_mn2_h100d2/no.out.mp4 2>&1 | tail -1)"; } | tee -a $L
log "v2 finished"
