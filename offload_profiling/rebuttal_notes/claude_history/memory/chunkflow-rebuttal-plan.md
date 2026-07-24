---
name: chunkflow-rebuttal-plan
description: NeurIPS 2026 rebuttal context for ChunkFlow paper (submission 20362) — reviewer concerns and experiment goals
metadata: 
  node_type: memory
  type: project
  originSessionId: 9dc698d9-3716-4725-8c28-61024b3e75fd
  modified: 2026-07-24T02:19:22.877Z
---

User (Han Meng, first author) is preparing the NeurIPS 2026 rebuttal for "ChunkFlow" (submission 20362, OpenReview 1RJwbtZKcG; paper=/workspace/sglang/preprint.pdf, reviews=/workspace/sglang/chunkflow_review.pdf). Ratings: gMFD 3 (conf 4), qYJy 2 (conf 5), iLgN 4 (conf 3); AC RCFG currently not inclined to accept.

AC's 5 major issues to address:
1. Validate on commodity PCIe GPUs (L40/A6000/L4-class), not just 2×H100-PCIe.
2. Scale beyond 2 GPUs (4/8-GPU PCIe nodes), other parallelism (TP / other SP variants).
3. Analytical model: clarify assumptions (T_comp ignores all-to-all; T_pref assumes exclusive PCIe), show it stays predictive with collectives at larger scale.
4. Show offloading is *necessary*: configs where no-offload OOMs (batch/frames/context beyond memory limits).
5. Generality beyond DiT (LLM/KV-cache discussion — appendix D exists; expand).

Experiment plan (user decisions): 5090-box only (no L40 rental); WanVideo only, Figure-3 arms only (no/old/new — NO ratio); launch comm-window mode (paper used it, not kernel); settings identical to paper (704×1280, 10 steps, frames 41–161, batch 1). Then SP=4 + OOM frontier.

SP=2 final (steady s/step no/old/new): f41 0.931/1.179/1.054, f81 2.060/2.213/2.182, f121 3.395/3.488/3.481, f161 5.190/5.192/5.213; peak alloc no→offload: 19.6→10.6 (f41), 23.1→14.1GB (f161). Convergence point left-shifted vs H100 exactly as F* model predicts (5090 P_peak ~3.6× lower). Repeatability ±0.5%. Data: offload_profiling/results_5090/sp2_f*/. Gotcha fixed: summarize sqlite cache was shared across configs (analyze_nsys.sh now passes per-config --export-dir).

Related: [[vast-5090-instance]]
