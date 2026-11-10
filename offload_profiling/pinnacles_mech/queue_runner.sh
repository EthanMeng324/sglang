#!/bin/bash
# Serial job feeder for the Pinnacles `test` partition (QOS allows one submitted job per user).
# Reads specs "<model> <cfg> <arms> [TAG] [KEY=VALUE,KEY=VALUE...]" from results/queue.txt, one per line
# (# = comment), submits the first one whenever no job of ours is in the test partition, then removes it.
# TAG and the extra KEY=VALUE pairs are passed to sb.sh as positional arguments (exported there).
cd /data/hanmeng/chunkflow || exit 1
echo "$(date) queue_runner started pid $$" >> results/jobs.txt
while true; do
    if squeue -h -u "$USER" -p test 2>/dev/null | grep -q .; then sleep 60; continue; fi
    spec=$(grep -v '^[[:space:]]*#' results/queue.txt 2>/dev/null | grep -v '^[[:space:]]*$' | head -1)
    if [ -z "$spec" ]; then sleep 60; continue; fi
    set -- $spec
    extra_env=()
    if [ -n "${5:-}" ]; then IFS=',' read -r -a extra_env <<<"$5"; fi
    if jid=$(sbatch --parsable sb.sh "$1" "$2" "$3" "TAG=${4:-}" "${extra_env[@]}" 2>>results/jobs.txt); then
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
