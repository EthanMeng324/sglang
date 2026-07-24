---
name: chunkflow-repo-facts
description: "Key non-obvious facts about the ChunkFlow sglang fork (branch v0.5.9-shard) — mode switches, gotchas, harness"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9dc698d9-3716-4725-8c28-61024b3e75fd
  modified: 2026-07-23T23:44:47.698Z
---

ChunkFlow fork (EthanMeng324/sglang, branch v0.5.9-shard, forked at bbe9c7eeb, +14.4k lines, all under python/sglang/multimodal_gen/ + offload_profiling/):

- **Mode switch is env-only**: `SGLANG_DIT_COMM_AWARE_OFFLOAD=1` selects `layerwise_offload_chunkwise.py` at IMPORT time (layerwise_offload.py:14); no CLI flag; forgetting it silently runs the old baseline. Chunk size: `SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB` (code default 32, paper C=16 — scripts always set 16).
- 4 modes: no (`--dit-layerwise-offload false`), old/Layerwise (offload on, COMM_AWARE=0), new/ChunkFlow (COMM_AWARE=1, WINDOW_MODE=kernel, chunk 16MB), ratio (+`SGLANG_DIT_OFFLOAD_RESIDENT_RATIO=0.4`, tensorwise prefix; PHASE_AWARE=0).
- Harness: `offload_profiling/run_profile.sh <model> [--num-frames N] [--steps warmup,no,old,new,ratio,analyze]` → run_access*.sh → nsys-wrapped `sglang generate`. **run_access*.sh + analyze_nsys.sh start with TAMU `module load` lines that abort on other machines — must strip.** Model paths default to /scratch/user/u.hm347392 — set `WANVIDEO_MODEL_PATH` etc. Legacy run.sh/run_no.sh/run_old.sh trio = 1-GPU mock-comm (SGLANG_WAN_MOCK_COMM_ENABLE=1) — NOT for real SP numbers.
- Step time metric: NVTX `SGL_DENOISING_STEP_i` via summarize_profile_matrix.py (drops step 0 + Wan2.2 switch-step outlier ≥1.15×median); peak mem from perf JSON (`--perf-dump-path`) memory_checkpoints. Output filenames don't encode SP degree — use separate RESULTS_DIR per degree.
- Wan2.2-TI2V-5B: 24 heads → SP degree 2/4 OK; HEIGHT must be 704 not 720. Paper settings: 704×1280, 10 steps, frames 41–161, guidance 4.0/3.0, `fa` attention backend (see below), torch.compile on, prompt "A cat walks on the grass, realistic".
- Attention backend: user confirmed paper runs did NOT use sageattention (despite script default sage_attn) — use `fa`, which on sm_120 auto-falls-back to Torch SDPA (cuda.py:296-300). CuTe-DSL `fused_scale_residual_norm_scale_shift` (every Wan block) has no arch fallback — smoke-test JIT on sm_120 first. `--dit-comm-active-window-mode`: paper numbers used `launch` (user confirmed); `kernel` is post-submission code.
- Under SP each rank pins a FULL CPU copy of DiT weights (~10GB×N ranks for Wan 5B).
- **We added** `SGLANG_WAN_VAE_TILING=1` env switch (configs/models/vaes/wanvae.py __post_init__) enabling tiled+temporal+parallel VAE decode — untiled Wan VAE decode OOMs on 32GB cards even at 21 frames (denoise itself fine). run_rebuttal.sh sets it for all runs so OOM frontier stays in the DiT phase. New clean driver: `offload_profiling/run_rebuttal.sh <SP> <FRAMES> [MODES]` (fa backend, launch window, results → /dev/shm/chunkflow_results + persisted summaries → offload_profiling/results_5090/).
- Wan pipeline defaults (seen in server_args dump): dit_cpu_offload=true (whole-model stage offload; auto-disabled when layerwise on), vae_cpu_offload=true — so "no-offload" arm still has weights resident during denoise, matching paper semantics.
- Key docs (Chinese): `python/sglang/multimodal_gen/runtime/prefetch_comm_contention_analysis.md` (analytical model + F* closed forms; H100 calib P_peak=756, BW=63GB/s), `comm_aware_trace_findings.md` (trace post-mortems; Flux flagged as noisy validation target; Wan/Hunyuan clean).

Related: [[chunkflow-rebuttal-plan]], [[vast-5090-instance]]
