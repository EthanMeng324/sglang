#!/bin/bash
# Diagnosis of the two H100 gate anomalies: groups A -> D -> B -> (C only on decision). One arm = one process via
# run_mn_arm_diag.sh (the copy). Every run has its own TAG directory; nothing existing is overwritten.
source /scratch/yx2432/rebuttal_mn/site.sh; source $WS/sglang/.venv/bin/activate
export NNODES=1 GPUS_PER_NODE=2 MASTER_ADDR=$(hostname -s) NCCL_DEBUG=WARN
G=$WS/diag; L=$G/diag_log.txt; RUN=$WS/run_mn_arm_diag.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a $L; }
log "diag start job=$SLURM_JOB_ID host=$(hostname -s) commit=$(cd $WS/sglang && git log --oneline -1 | cut -c1-9)"
arm() { # frames arm TAG [VAR=VAL ...]
  local f=$1 a=$2 tag=$3; shift 3
  local out=$WS/results/wan_c${f}_mn2_${tag}
  if grep -q "average time per step" $out/logs/$a.log 2>/dev/null; then log "skip $f $a $tag (has result)"; return; fi
  mkdir -p $out/logs; nvidia-smi dmon -s pc -d 1 -o T > $out/logs/dmon_$a.txt 2>&1 & local dm=$!
  log "run f=$f arm=$a TAG=$tag $*"
  env TAG=$tag "$@" bash $RUN wan $f $a 7; local rc=$?
  kill $dm 2>/dev/null; wait $dm 2>/dev/null
  log "end f=$f arm=$a TAG=$tag rc=$rc $(grep -oE 'average time per step: [0-9.]+' $out/logs/$a.log | tail -1) files: $(ls $out | grep -E "^$a\.out" | tr '\n' ' ')"
}
# A: reproducibility of anomaly 2 (f161, interleaved)
arm 161 no h100d1 CHUNK_MB=16; arm 161 gate h100d1 CHUNK_MB=16; arm 161 no h100d2 CHUNK_MB=16; arm 161 gate h100d2 CHUNK_MB=16
v() { ls $WS/results/wan_c161_mn2_$1/$2.out* 2>/dev/null | head -1; }
{ echo "== PSNR (diag/psnr.py)"
  for pr in "h100d1 no h100d2 no" "h100d1 no h100d1 gate" "h100d2 no h100d2 gate"; do set -- $pr
    echo "$2_$1 vs $4_$3: $(python $G/psnr.py $(v $1 $2) $(v $3 $4) 2>&1 | tail -1)"; done; } | tee -a $L
# D: chunk size for anomaly 1 (f41)
arm 41 gate h100c32 CHUNK_MB=32; arm 41 gate h100c64 CHUNK_MB=64
# B: nsys f161 no and gate
arm 161 no h100n CHUNK_MB=16 NSYS=1; arm 161 gate h100n CHUNK_MB=16 NSYS=1
B=$WS/results/wan_c161_mn2_h100n
log "nsys analysis"
python $WS/sglang/offload_profiling/analyze_nsys_step_metrics.py $B/profiles/*_nsys.nsys-rep --output $B/step_metrics.md --export-dir $B/profiles/sqlite_cache >> $L 2>&1; log "analyze rc=$?"
ls $B/profiles/sqlite_cache | tee -a $L
q() { ls $B/profiles/sqlite_cache/$1_nsys*.sqlite 2>/dev/null | head -1; }
python $G/kernels.py 5 $(q no) $(q gate) > $B/kernels_step5.md 2>>$L; log "kernels rc=$? -> $B/kernels_step5.md"; head -6 $B/kernels_step5.md | tee -a $L
# C: only on decision (written by the operator after reading B): diag/C_DECISION = run | skip
log "waiting for diag/C_DECISION (run|skip), up to 60 min"
for i in $(seq 1 240); do [ -s $G/C_DECISION ] && break; sleep 15; done
d=$(cat $G/C_DECISION 2>/dev/null); log "C decision: ${d:-none (timeout) -> skip}"
if [ "$d" = run ]; then arm 161 no h100e CHUNK_MB=16 COMPILE=0; arm 161 gate h100e CHUNK_MB=16 COMPILE=0; fi
log "diag finished"
