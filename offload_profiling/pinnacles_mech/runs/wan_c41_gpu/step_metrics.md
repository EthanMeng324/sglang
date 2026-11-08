# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c41_gpu/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_gpu/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_gpu/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gpu_nsys | 1557.518 | 657.839 | 185.655 | 184.156 | 473.643 | 9 | 2 | 73 |
| launch_nsys | 1339.783 | 648.627 | 478.403 | 3.491 | 204.149 | 9 | 2 | 73 |
| no_nsys | 1142.193 | 458.341 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
