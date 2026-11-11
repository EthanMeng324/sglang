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
