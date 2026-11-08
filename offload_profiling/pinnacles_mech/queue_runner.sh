#!/bin/bash
# Serial job feeder for the Pinnacles `test` partition (QOS allows one submitted job per user).
# Reads specs "<model> <cfg> <arms> [TAG]" from results/queue.txt, one per line (# = comment),
# submits the first one whenever no job of ours is in the test partition, then removes it.
cd /data/hanmeng/chunkflow || exit 1
echo "$(date) queue_runner started pid $$" >> results/jobs.txt
while true; do
    if squeue -h -u "$USER" -p test 2>/dev/null | grep -q .; then sleep 60; continue; fi
    spec=$(grep -v '^[[:space:]]*#' results/queue.txt 2>/dev/null | grep -v '^[[:space:]]*$' | head -1)
    if [ -z "$spec" ]; then sleep 60; continue; fi
    set -- $spec
    if jid=$(TAG="${4:-}" sbatch --parsable sb.sh "$1" "$2" "$3" 2>>results/jobs.txt); then
        echo "$jid" > results/current_job
        echo "$(date) submitted $jid: $spec" >> results/jobs.txt
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
