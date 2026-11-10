# Multi-node (2 × 4 L40S) offload runbook — ChunkFlow rebuttal experiment

Goal: the reviewers asked for the offload comparison at 8 GPUs. The L40S nodes we have hold at most
4 GPUs, so the 8-GPU run is 2 nodes × 4 GPUs with one Ulysses sequence-parallel group (SP = 8) across
both nodes. First pass: two arms only, **`no`** (no offload) and **`old`** (the paper's "Layerwise"
baseline), on Wan2.2-TI2V-5B at 704×1280, 41 and 81 frames, 10 denoising steps, torch.compile on.
Everything below also works for a 4-GPU single-node run (`NNODES=1 GPUS_PER_NODE=4`), which is the
same-hardware reference the 8-GPU numbers are compared against.

Code: branch **`pinnacles-mech`** of `https://github.com/EthanMeng324/sglang` (commit with
"Multi-node launch" in its message or later). The harness lives in `offload_profiling/pinnacles_mech/`.

## 1. What the code does across nodes

- `sglang generate --num-gpus 8 --ulysses-degree 8 --nnodes 2 --node-rank K --dist-init-addr HOST:PORT ...`
  is run **once per node, at the same time, with identical arguments except `--node-rank`**.
- `--num-gpus` is the total. Node K launches `num_gpus / nnodes` worker processes with global ranks
  `K*4 + i`, all joining one `torch.distributed` group initialised at `HOST:PORT` (node 0).
- Node 0 hosts the scheduler and sends the request; it reaches every rank through a gloo broadcast.
  Node 1 (`--node-rank 1`) only hosts workers and exits when node 0 broadcasts the shutdown. If node 0
  dies early, node 1's processes block in the distributed init / broadcast — always run node 1 under a
  timeout (the harness does; `srun --wait=90` on Slurm).
- Weights are prefetched per GPU from that GPU's host memory over its own PCIe link; the all-to-all
  of Ulysses crosses the inter-node network for half of its traffic. So the inter-node link quality
  matters: record it (step 3) before reading any result.

## 2. Environment (per node; a shared filesystem for the repo, venv and models is simplest)

Versions that work on L40S: Python 3.12, `torch 2.9.1+cu128`, `sglang 0.5.5.post3` (this repo,
editable), `sgl-kernel` (FlashAttention backend `fa`), CUDA 12.8/12.9 toolkit on `CUDA_HOME` for the
JIT pieces (flashinfer rope, inductor). Models (HF): `Wan-AI/Wan2.2-TI2V-5B-Diffusers`,
`hunyuanvideo-community/HunyuanVideo` (optional), `black-forest-labs/FLUX.1-dev` (optional, gated).

```bash
git clone -b pinnacles-mech https://github.com/EthanMeng324/sglang.git
cd sglang && python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e "python[all]"  # or: pip install -e python && pip install sgl-kernel
export CC=gcc CXX=g++      # NOT nvc; do not `module load nvhpc`
export CUDA_HOME=/path/to/cuda-12.x   # must contain bin/nvcc and lib64/libcudart.so
mkdir -p models && huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Wan2.2-TI2V-5B-Diffusers
sglang generate --help | grep -E "nnodes|node-rank|dist-init-addr"   # the multi-node flags must be listed
```

The harness scripts assume `WS=/data/hanmeng/chunkflow` with `WS/sglang`, `WS/models/...`,
`WS/results`; set `WS=/your/path` in the environment (every script honours it) or edit the top of
`run_mn_arm.sh`. Caches: `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` on a shared path so the second
node compiles once.

## 3. Before measuring: topology and the inter-node link

On each node: `nvidia-smi topo -m`, `nvidia-smi --query-gpu=name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv`,
`ls /sys/class/infiniband` + `ibstat` (or `ibv_devinfo`), `ip -br link`. Record whether the GPUs sit on a
PCIe switch or directly on the CPU root ports, and whether the nodes talk over InfiniBand/RoCE or Ethernet
(and its speed).

NCCL cross-node bandwidth (informational, 1 min): run on both nodes at once
```bash
torchrun --nnodes 2 --node_rank $K --master_addr $HOST --master_port 29399 --nproc_per_node 4 offload_profiling/pinnacles_mech/nccl_bw.py
```
It prints per-rank all_to_all / send-recv GB/s at 4–256 MB. Single-node 2×L40S (PCIe Gen4 ×16, no
switch) gives all_to_all ≈ 14 GB/s per rank; a 100 Gb/s link gives ≈ 12 GB/s per NIC in total, so with
4 GPUs sharing one NIC expect the cross-node all_to_all to be several times slower than intra-node. If
NCCL picks the wrong interface set `NCCL_SOCKET_IFNAME=<ib or 100GbE iface>` (`GLOO_SOCKET_IFNAME` likewise)
and `NCCL_IB_HCA=mlx5_0` (adjust); `NCCL_DEBUG=INFO` shows the transport (`NET/IB` vs `NET/Socket`).

## 4. Running the arms

`run_mn_arm.sh <model> <cfg> <arm> [port_offset]` runs ONE arm on ONE node; launch it on every node at
the same moment with the same arguments. Required env: `MASTER_ADDR` (node 0's hostname/IP as seen
from the other nodes), `NNODES`, `GPUS_PER_NODE`, `NODE_RANK` (taken from `$SLURM_NODEID` when under
Slurm). Optional: `STEPS=10`, `NSYS=0|1`, `TAG`, `OUT`.

Slurm (two nodes in one allocation; this is what `sb_mn.sh` does on Pinnacles, adjust the `#SBATCH` lines):
```bash
export WS=/your/ws NNODES=2 GPUS_PER_NODE=4 MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
for arm in no old; do
  srun -N 2 --ntasks-per-node=1 --wait=90 --kill-on-bad-exit=0 bash $WS/run_mn_arm.sh wan 41 $arm 7
done
```
Without Slurm: on node 0 `NODE_RANK=0 bash run_mn_arm.sh wan 41 no 7` and on node 1
`NODE_RANK=1 bash run_mn_arm.sh wan 41 no 7` (start node 1 within a few seconds of node 0; both must use
the same `MASTER_ADDR`, `port_offset`, model/cfg/arm). Then the next arm.

Arms (same env mapping as the single-node harness):
- `no`: `--dit-layerwise-offload false` — No Offload.
- `old`: `SGLANG_DIT_OFFLOAD_COMM_QUIESCE=1`, `--dit-layerwise-offload true --dit-offload-prefetch-size 1` —
  the paper's "Layerwise" baseline (whole-layer prefetch, copy stream drained before every collective).
- `stock` (optional, free): same with `SGLANG_DIT_OFFLOAD_COMM_QUIESCE=0` — upstream SGLang behaviour.
- `gate` (optional): the comm-aware chunked prefetch with the GPU-side gate (`CHUNK_MB=4`).

Configs for the first pass: `wan 41`, `wan 81` (both 704×1280). Then, if time allows, `hunyuan 13`
(720p). Run each config at 8 GPUs (2×4) **and** at 4 GPUs on one node (`NNODES=1 GPUS_PER_NODE=4`,
same script, `--node-rank 0` only) so the 4→8 scaling of both arms is on the same hardware.

## 5. What to collect and report

Per (config, GPUs, arm): from node 0's log `OUT/logs/<arm>.log`
- `[DenoisingStage] average time per step: X seconds` (10 steps; step 0 includes compile — the harness
  also logs per-step times with `SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES=1`; report the mean of steps 1–9
  if the per-step lines are present, else the stage average),
- `[DenoisingStage] finished in Y seconds`, end-to-end wall time, and peak GPU memory from
  `OUT/profiles/perf_<arm>.json` (`peak_memory_mb` / memory snapshots),
- the NCCL transport line (`NCCL INFO ... via NET/IB` or `NET/Socket`) from any worker log,
- `OUT/logs/nodes.txt` and `OUT/logs/nccl_bw.txt` (topology and cross-node bandwidth from step 3).

Report as a table: config | GPUs (nodes×per-node) | arm | steady step ms | old/no ratio | peak GB | notes.
Run every cell at least twice; the single-node runs repeat within 1%, cross-node runs may not — report
both values if they differ by more than 3%.

If `NSYS=1` is set, each node writes `OUT/profiles/<arm>[_nodeK]_nsys.nsys-rep`; `analyze_mech.sh`
and `link_budget.py` in this directory read node 0's report (rank 0 is in it).

## 6. Known failure modes

- Node 1 stuck at startup: `--dist-init-addr` host not reachable from node 1 (use the IP of the
  interface that routes between nodes), or port in use — the run uses `29500 + port_offset`.
- `NCCL ... NET/Socket` over the management Ethernet: set `NCCL_SOCKET_IFNAME` to the fast interface.
- Compile cache contention: keep `TORCHINDUCTOR_CACHE_DIR` per node or on a shared FS; the first arm
  takes a few minutes longer (compile).
- Out of memory in the `no` arm at 81 frames on 46 GB: that is a result (report it), the `old` arm
  still runs; the VAE decode may OOM after the denoising stage — the step times before it are valid.
