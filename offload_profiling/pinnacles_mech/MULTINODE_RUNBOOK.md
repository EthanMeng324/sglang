# Runbook — WanVideo 4-point sweep, No Offload vs Layerwise, on (a) 2 × 4 L40S and (b) 1 × 2 H100 NVLink

Code: branch **`pinnacles-mech`** of `https://github.com/EthanMeng324/sglang` (`git pull` before the first run).
Harness: `offload_profiling/pinnacles_mech/run_mn_arm.sh` (one arm = one full process: load, compile, 10 steps,
decode; ~4 min per arm on 2×L40S). Both experiments use the same script; only the node count and GPU count differ.

## 1. The sweep

WanVideo `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, 704×1280, frames **41, 81, 121, 161**, 10 denoising steps, guidance
4.0/3.0, prompt "A cat walks on the grass, realistic", torch.compile on, FlashAttention backend `fa`, Ulysses
sequence parallelism over all GPUs of the run. Two arms, run once each:

- `no`  — No Offload: `--dit-layerwise-offload false`
- `old` — the paper's "Layerwise" baseline: `--dit-layerwise-offload true --dit-offload-prefetch-size 1`,
  `SGLANG_DIT_OFFLOAD_COMM_QUIESCE=1`

(`run_mn_arm.sh` sets these from the arm name.) 4 frame counts × 2 arms = 8 arms per platform.

Platforms:
- **(a) 8 GPUs = 2 nodes × 4 L40S**, SP = 8 across the two nodes (`NNODES=2 GPUS_PER_NODE=4`). Node 0 drives
  the request and receives the result; node 1 only hosts workers.
- **(b) 2 × H100 with NVLink, one node**, SP = 2 (`NNODES=1 GPUS_PER_NODE=2`). This is the NVLink control: the
  all-to-all stays on NVLink, only the prefetch uses PCIe.

Run (b) while waiting for the two L40S nodes; run (a) as soon as the allocation comes.

## 2. Environment (per node; a shared filesystem for repo, venv and models is simplest)

Python 3.12, `torch 2.9.1+cu128`, this repo editable (`pip install -e python`), `sgl-kernel`, a CUDA 12.8/12.9
toolkit on `CUDA_HOME` (JIT: flashinfer rope, inductor), `CC=gcc CXX=g++` (not nvc).

```bash
git clone -b pinnacles-mech https://github.com/EthanMeng324/sglang.git && cd sglang
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e python && pip install sgl-kernel
export CC=gcc CXX=g++ CUDA_HOME=/path/to/cuda-12.x
mkdir -p models && huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Wan2.2-TI2V-5B-Diffusers
sglang generate --help | grep -E "nnodes|node-rank|dist-init-addr"   # the multi-node flags must be listed
```

`run_mn_arm.sh` expects `WS` with `WS/sglang` (this repo), `WS/models/Wan2.2-TI2V-5B-Diffusers`, `WS/results`:
`export WS=/your/workspace` (or edit the top of the script). Put `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`
on a shared path so every node reuses one compile cache.

## 3. Running one arm

`bash run_mn_arm.sh wan <frames> <no|old> [port_offset]`, started **on every node of the run at the same time**
with identical arguments; env: `NNODES`, `GPUS_PER_NODE`, `NODE_RANK` (= `$SLURM_NODEID` under Slurm),
`MASTER_ADDR` (node 0's hostname/IP reachable from the other node; unused when `NNODES=1`), optional `TAG`,
`OUT`, `NSYS=0` (default; leave it off).

(b) one node, 2 × H100:
```bash
export WS=/your/ws NNODES=1 GPUS_PER_NODE=2 TAG=h100
for f in 41 81 121 161; do for arm in no old; do bash $WS/sglang/offload_profiling/pinnacles_mech/run_mn_arm.sh wan $f $arm 7; done; done
```
(a) two nodes, 4 × L40S each, under Slurm (one `srun` per arm; `--wait=90` kills the worker-only node 90 s after
node 0 exits, so a crashed node 0 cannot leave node 1 hanging):
```bash
export WS=/your/ws NNODES=2 GPUS_PER_NODE=4 TAG=l40s8 MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
for f in 41 81 121 161; do for arm in no old; do
  srun -N 2 --ntasks-per-node=1 --wait=90 --kill-on-bad-exit=0 bash $WS/sglang/offload_profiling/pinnacles_mech/run_mn_arm.sh wan $f $arm 7
done; done
```
Without Slurm: run the same command by hand on node 0 (`NODE_RANK=0`) and node 1 (`NODE_RANK=1`) within a few
seconds of each other, same `MASTER_ADDR`, port offset and arguments; wrap node 1 in `timeout`.

Results: `OUT = $WS/results/wan_c<frames>_mn<total_gpus>_<TAG>/`, node 0 log `logs/<arm>.log`, node 1 log
`logs/<arm>_node1.log`, `logs/run.txt` (one line per arm with rc and the stage average), `profiles/perf_<arm>.json`.

## 4. What to record per (platform, frames, arm)

- steady step time: the per-step lines in the rank-0 log (`SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES=1` is set by
  the script) — mean of steps 1–9 (step 0 includes compile); if the per-step lines are missing, the stage line
  `[DenoisingStage] average time per step: X seconds`;
- `[DenoisingStage] finished in Y seconds` and the end-to-end wall time from `run.txt`;
- peak GPU memory from `profiles/perf_<arm>.json` (`peak_memory_mb` / memory snapshots);
- the interconnect actually used: once per platform `nvidia-smi topo -m` (NV# links between the two H100s;
  `NODE`/`SYS` on the L40S nodes) and one NCCL line from any worker log — `NCCL INFO ... via P2P/...` on
  NVLink, `via NET/IB` or `via NET/Socket` between the L40S nodes (set `NCCL_DEBUG=INFO` for one arm if the
  default `WARN` level prints nothing).

Table: platform | frames | arm | steady step ms | old/no ratio | peak GB | interconnect | notes (OOM etc.).
Attach `run.txt`, the rank-0 logs and `nvidia-smi topo -m` output.

## 5. Failure modes

- node 1 blocked at startup: `MASTER_ADDR` not reachable from node 1 (use the IP of the interface that routes
  between the nodes) or the port (`29500 + offset`) in use;
- NCCL over the management Ethernet (`NET/Socket` although there is InfiniBand): `NCCL_SOCKET_IFNAME=<ib or fast
  interface>`, `GLOO_SOCKET_IFNAME` likewise, `NCCL_IB_HCA=mlx5_0` (adjust);
- the first arm of a platform takes a few minutes longer (compile);
- an out-of-memory in the `no` arm at the large frame counts is a result: report it and continue; the VAE
  decode may OOM after the denoising stage — the step times before it are valid.
