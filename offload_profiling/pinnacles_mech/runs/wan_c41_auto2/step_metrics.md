# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c41_auto2/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto2/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto2/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto2/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 1463.458 | 670.158 | 331.392 | 95.452 | 234.797 | 9 | 2 | 73 |
| gpu_nsys | 1569.390 | 675.552 | 184.912 | 203.581 | 480.964 | 9 | 2 | 73 |
| launch_nsys | 1348.912 | 654.181 | 479.804 | 4.631 | 215.355 | 9 | 2 | 73 |
| no_nsys | 1144.196 | 458.631 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
