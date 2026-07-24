---
name: vast-5090-instance
description: "Hardware/env facts of the current Vast.ai box (4x RTX 5090, NOT L40 as user assumed; tight disk, huge RAM/shm)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9dc698d9-3716-4725-8c28-61024b3e75fd
  modified: 2026-07-23T23:58:13.802Z
---

Vast.ai instance used for ChunkFlow rebuttal experiments (as of 2026-07-23):
- **4× RTX 5090 32GB** (Blackwell sm_120), PCIe Gen5 x16, all on one NUMA node, **no NVLink, P2P disabled (CNS)** → pure PCIe-contention topology, ideal for the paper's data-path-sharing regime. User initially believed it was 4×L40.
- Driver 580.159.03 / CUDA 13.0; needs cu128+ wheels (sm_120).
- Host RAM 1TB (cgroup limit 734GB), /dev/shm 251GB, 192 cores.
- **Disk only 32GB total (~31GB free), /workspace is NOT a volume** (lost on recycle/destroy). Wan2.2-TI2V-5B-Diffusers repo = 34GB (transformer 20GB fp32, umt5 11GB, VAE 3GB) → must put HF cache in /dev/shm (lost on restart; re-download needed).
- /venv/main initially had NO torch/sglang — we installed /workspace/sglang/.venv (torch 2.9.1+cu128, sgl-kernel 0.3.21, flashinfer 0.6.3; ~11GB).
- **Trap**: base image exports `HF_HOME=/workspace/.hf_home` in ambient env — a `${HF_HOME:-...}` fallback never fires, and any run without explicit HF_HOME re-downloads 34GB onto the 32GB disk → disk-full crash cascade (worker EOFError + nsys "0 MiB free" ProtobufSerializeException). Always `export HF_HOME=/dev/shm/hf` unconditionally; also `TMPDIR=/dev/shm/nsys-tmp` for nsys intermediates. Bash-tool ENOSPC deadlock (can't even write tool output) can be broken via the Monitor tool whose command still runs.

Related: [[chunkflow-rebuttal-plan]]
