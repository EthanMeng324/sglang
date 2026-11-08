# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/hunyuan_c13_gpu/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_gpu/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_gpu/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gpu_nsys | 1751.003 | 463.641 | 42.290 | 265.265 | 404.375 | 9 | 2 | 81, 85, 7 |
| launch_nsys | 1703.732 | 537.021 | 373.192 | 136.940 | 59.974 | 9 | 2 | 81, 85, 7 |
| no_nsys | 1389.265 | 389.837 | 0.000 | 0.168 | 0.000 | 9 | 2 | 7 |
