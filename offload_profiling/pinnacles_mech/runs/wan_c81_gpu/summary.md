# wan_c81_gpu

| arm | step ms | vs no | NCCL ms | NCCL/H2D overlap ms | uncovered H2D ms | WAIT_COMM_BG ms | peak reserved GB | peak alloc GB | e2e s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no | 2396.8 | +0.0% | 833.4 | 0.0 | 0.0 | 0.0 | 29.22 | 23.00 | 56.11 |
| launch | 2561.5 | +6.9% | 996.8 | 465.5 | 1.8 | 255.3 | 17.34 | 14.15 | 55.18 |
| gpu | 2439.2 | +1.8% | 865.7 | 127.9 | 4.0 | 455.7 | 17.35 | 14.21 | 54.39 |
