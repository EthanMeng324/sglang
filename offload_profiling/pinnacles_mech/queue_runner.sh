#!/bin/bash
# Serial job feeder for the Pinnacles `test` partition (QOS allows one submitted job per user).
# Reads specs "<model> <cfg> <arms> [TAG] [KEY=VALUE,KEY=VALUE...]" from results/queue.txt, one per line
# (# = comment), submits the first one whenever no job of ours is in the test partition, then removes it.
# TAG and the extra KEY=VALUE pairs are passed to the sbatch script as positional arguments (exported there).
# Model prefixes select the sbatch script: "mn:wan" -> sb_mn.sh (2 nodes x 2 L40S), "mn4:wan" -> sb_mn4.sh
# (2 nodes x 4 L40S); no prefix -> sb.sh (1 node x 2 L40S).
# Run exactly ONE instance (there are two login nodes behind login.rc.ucmerced.edu; check both).
cd /data/hanmeng/chunkflow || exit 1
echo "$(date) queue_runner started pid $$ on $(hostname)" >> results/jobs.txt
while true; do
    if squeue -h -u "$USER" -p test 2>/dev/null | grep -q .; then sleep 60; continue; fi
    spec=$(grep -v '^[[:space:]]*#' results/queue.txt 2>/dev/null | grep -v '^[[:space:]]*$' | head -1)
    if [ -z "$spec" ]; then sleep 60; continue; fi
    set -- $spec
    model="$1"; script=sb.sh
    case "$model" in
        mn4:*) script=sb_mn4.sh; model="${model#mn4:}" ;;
        unit:*) script=sb_unit.sh; model="${model#unit:}" ;;
        g1:*)  script=sb_1g.sh;  model="${model#g1:}" ;;
        mn1:*) script=sb_mn1.sh; model="${model#mn1:}" ;;
        mn:*)  script=sb_mn.sh;  model="${model#mn:}" ;;
    esac
    extra_env=()
    if [ -n "${5:-}" ]; then IFS=',' read -r -a extra_env <<<"$5"; fi
    if jid=$(sbatch --parsable "$script" "$model" "$2" "$3" "TAG=${4:-}" "${extra_env[@]}" 2>>results/jobs.txt); then
        echo "$jid" > results/current_job
        echo "$(date) submitted $jid ($script): $spec" >> results/jobs.txt
        # drop the first non-comment, non-empty line
        python3 - <<'PY'
import pathlib
p = pathlib.Path("results/queue.txt"); lines = p.read_text().splitlines(True); out = []; dropped = False
for l in lines:
    if not dropped and l.strip() and not l.lstrip().startswith("#"):
        dropped = True; continue
    out.append(l)
p.write_text("".join(out))
PY
        sleep 120
    else
        echo "$(date) submit failed: $spec" >> results/jobs.txt
        sleep 120
    fi
done
