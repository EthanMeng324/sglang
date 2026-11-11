# (b) 1 node x 2 H100 80GB HBM3 (NVLink NV6), SP=2 — WanVideo Wan2.2-TI2V-5B, 704x1280, 10 steps, torch.compile, fa
job 22596821 on gh007 (partition h100_tandon), 2026-11-10 18:20-18:59, branch pinnacles-mech @ 864c48c0f
interconnect: nvidia-smi topo -m GPU0-GPU1 = NV6; NCCL: "Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM" (no-arm, 41 frames)
steady step = mean of steps 1-9 from profiles/perf_<arm>.json denoise_steps_ms (same per-step timer the log uses)

| platform | frames | arm | steady step ms | old/no | peak GB (alloc / reserved) | DenoisingStage finished s | wall s | interconnect | notes |
|---|---|---|---|---|---|---|---|---|---|
| 1x2 H100 | 41  | no  | 385  |      | 23.0 / 27.1 | 118.3 | 435 | NVLink NV6, P2P/CUMEM | first arm: step 0 = 114 s (cold compile cache) |
| 1x2 H100 | 41  | old | 700  | 1.82 | 14.2 / 18.0 | 39.5  | 295 | NVLink NV6 | step 9 = 1146 ms (steps 1-8 = 644) |
| 1x2 H100 | 81  | no  | 723  |      | 23.5 / 29.3 | 74.9  | 262 | NVLink NV6 | |
| 1x2 H100 | 81  | old | 1029 | 1.42 | 14.6 / 21.0 | 43.4  | 246 | NVLink NV6 | step 9 = 1318 ms (steps 1-8 = 993) |
| 1x2 H100 | 121 | no  | 1221 |      | 23.9 / 30.6 | 85.3  | 281 | NVLink NV6 | |
| 1x2 H100 | 121 | old | 1451 | 1.19 | 15.0 / 19.3 | 44.9  | 256 | NVLink NV6 | step 9 = 1500 ms |
| 1x2 H100 | 161 | no  | 1870 |      | 24.3 / 29.3 | 84.5  | 294 | NVLink NV6 | |
| 1x2 H100 | 161 | old | 2041 | 1.09 | 15.5 / 19.7 | 50.9  | 261 | NVLink NV6 | |

Every arm rc=0, no OOM. All logs contain "torch._dynamo hit config.recompile_limit (16)" for
torch_dynamo_resume_in_forward_at_503 (reason: L['mock_comm_fn'] changed) — inherited from the harness, not changed.
Stage "average time per step" includes step 0 (compile), so it is not the steady number.
Files: wan_c<f>_mn2_h100/logs/{run.txt,no.log,old.log}, wan_c<f>_mn2_h100/profiles/perf_<arm>.json, topo_h100.txt, driver_h100.txt

## Added 2026-11-11: stock and gate arms (job 22737983, gh015, h100_tandon, same node type and settings)
No/old rows below are the same runs as above (job 22596821), not re-run. CHUNK_MB=16 exported by the driver
(run_mn_arm.sh default is 4). Env lines in run.txt: stock has SGLANG_DIT_OFFLOAD_COMM_QUIESCE=0; gate has
SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB=16 and SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE=gate. gate logs show
"[comm-gate] stream memops gate ready on cuda:0/1" and "chunk=16MB ... comm_window=gate"; no memops errors.
Every arm rc=0. Steps are perf_<arm>.json denoise_steps_ms as recorded; arm/no uses the matching mean.
At 81-161 frames gate is faster than no (0.92-0.98) and stock is within 2% of no; reported as measured.
Driver change for this run: arm list per platform (h100: no old stock gate; l40s8 stays no old) + CHUNK_MB=16.

| platform | frames | arm | steps 0-9 ms (denoise_steps_ms) | mean 1-9 ms | mean 1-8 ms | arm/no (1-9) | arm/no (1-8) | peak GB alloc / reserved | rc | wall s | interconnect | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x2 H100 | 41 | no | 114273, 384, 385, 383, 384, 384, 384, 384, 383, 389 | 385 | 384 |  |  | 23.0 / 27.1 | 0 | 435 | NVLink NV6 | step 0 cold compile cache |
| 1x2 H100 | 41 | old | 33144, 643, 644, 645, 644, 645, 644, 644, 644, 1145 | 700 | 645 | 1.82 | 1.68 | 14.2 / 18.0 | 0 | 295 | NVLink NV6 | step 9 slow |
| 1x2 H100 | 41 | stock | 36220, 418, 412, 413, 411, 411, 412, 414, 421, 416 | 415 | 414 | 1.08 | 1.08 | 14.2 / 18.0 | 0 | 282 | NVLink NV6 |  |
| 1x2 H100 | 41 | gate | 29106, 476, 470, 473, 475, 473, 474, 477, 547, 581 | 494 | 484 | 1.28 | 1.26 | 14.2 / 17.4 | 0 | 265 | NVLink NV6 | step 9 slow; memops gate ready, chunk=16MB |
| 1x2 H100 | 81 | no | 68286, 719, 725, 723, 725, 724, 722, 721, 724, 723 | 723 | 723 |  |  | 23.5 / 29.3 | 0 | 262 | NVLink NV6 |  |
| 1x2 H100 | 81 | old | 34085, 998, 993, 991, 991, 991, 992, 991, 992, 1317 | 1029 | 993 | 1.42 | 1.37 | 14.6 / 21.0 | 0 | 246 | NVLink NV6 | step 9 slow |
| 1x2 H100 | 81 | stock | 31123, 721, 725, 723, 725, 726, 726, 727, 725, 726 | 725 | 725 | 1.00 | 1.00 | 14.6 / 18.0 | 0 | 263 | NVLink NV6 |  |
| 1x2 H100 | 81 | gate | 31446, 729, 714, 706, 702, 701, 702, 704, 704, 707 | 708 | 708 | 0.98 | 0.98 | 14.6 / 17.4 | 0 | 258 | NVLink NV6 | memops gate ready, chunk=16MB |
| 1x2 H100 | 121 | no | 74194, 1233, 1219, 1217, 1221, 1214, 1220, 1220, 1219, 1219 | 1221 | 1221 |  |  | 23.9 / 30.6 | 0 | 281 | NVLink NV6 |  |
| 1x2 H100 | 121 | old | 31758, 1446, 1444, 1444, 1445, 1445, 1442, 1444, 1445, 1500 | 1451 | 1445 | 1.19 | 1.18 | 15.0 / 19.3 | 0 | 256 | NVLink NV6 |  |
| 1x2 H100 | 121 | stock | 28966, 1220, 1222, 1221, 1227, 1226, 1223, 1226, 1225, 1399 | 1244 | 1224 | 1.02 | 1.00 | 15.0 / 22.4 | 0 | 262 | NVLink NV6 |  |
| 1x2 H100 | 121 | gate | 27707, 1214, 1194, 1179, 1182, 1173, 1177, 1174, 1173, 1175 | 1183 | 1184 | 0.97 | 0.97 | 15.0 / 18.7 | 0 | 261 | NVLink NV6 | memops gate ready, chunk=16MB |
| 1x2 H100 | 161 | no | 67639, 1872, 1868, 1870, 1871, 1869, 1869, 1869, 1869, 1872 | 1870 | 1870 |  |  | 24.3 / 29.3 | 0 | 294 | NVLink NV6 |  |
| 1x2 H100 | 161 | old | 32459, 2037, 2041, 2039, 2041, 2040, 2041, 2040, 2042, 2045 | 2041 | 2041 | 1.09 | 1.09 | 15.5 / 19.7 | 0 | 261 | NVLink NV6 |  |
| 1x2 H100 | 161 | stock | 31261, 1861, 1869, 1869, 1866, 1865, 1872, 1874, 1872, 1870 | 1869 | 1869 | 1.00 | 1.00 | 15.5 / 19.7 | 0 | 280 | NVLink NV6 |  |
| 1x2 H100 | 161 | gate | 32004, 1839, 1782, 1707, 1701, 1699, 1700, 1698, 1700, 1698 | 1725 | 1729 | 0.92 | 0.92 | 15.5 / 19.1 | 0 | 284 | NVLink NV6 | memops gate ready, chunk=16MB |
