# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c81_gpu/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_gpu/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_gpu/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gpu_nsys | 2439.165 | 865.674 | 127.885 | 4.040 | 455.722 | 9 | 2 | 73 |
| launch_nsys | 2561.490 | 996.792 | 465.500 | 1.780 | 255.272 | 9 | 2 | 73 |
| no_nsys | 2396.759 | 833.373 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
