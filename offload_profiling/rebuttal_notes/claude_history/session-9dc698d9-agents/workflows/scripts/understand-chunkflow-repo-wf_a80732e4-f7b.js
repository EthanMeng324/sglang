export const meta = {
  name: 'understand-chunkflow-repo',
  description: 'Parallel deep-read of the ChunkFlow sglang fork: offloading runtime, experiment harness, docs, multi-GPU readiness',
  phases: [
    { title: 'Read', detail: 'four parallel readers over the repo' },
  ],
}

const SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'Detailed findings, structured markdown' },
    key_files: { type: 'array', items: { type: 'string' }, description: 'repo-relative paths of the most important files with 1-line notes appended after a colon' },
    caveats: { type: 'array', items: { type: 'string' }, description: 'gotchas, missing pieces, or risks relevant to running experiments' },
  },
  required: ['summary', 'key_files', 'caveats'],
}

const CTX = `Context: /workspace/sglang is a fork of SGLang (branch v0.5.9-shard) implementing "ChunkFlow" — communication-aware chunked prefetching for layerwise weight offloading in distributed diffusion transformer (DiT) inference (paper: preprint.pdf in repo root; do NOT read the PDFs, they are already read). Key paper concepts: layerwise offloading with chunked (pauseable/resumable) H2D prefetch that yields to NCCL collectives (all-to-all for Ulysses SP) at chunk boundaries via a pause flag + CUDA event; partial parameter residency (keep a fraction of chunks resident); chunk size C=16MB default; evaluated on WanVideo (Wan2.2-TI2V-5B), Flux, HunyuanVideo with Ulysses SP degree 2 on 2xH100. The user now needs to re-run experiments on commodity GPUs (this box: 4x RTX 5090 32GB, CUDA 13 driver, Blackwell sm_120) at SP degree 2 and 4, for a NeurIPS rebuttal.`

const TASKS = [
  {
    key: 'runtime',
    prompt: `${CTX}

Your task: find and explain the ChunkFlow offloading RUNTIME implementation in this fork. Search under python/sglang (especially anything related to multimodal generation, diffusion, offload, prefetch, chunk, residency, pause/resume, comm window). Recent commits mention "comm-window-mode kernel", "kernel end marker", "monitor work complete marker" — find what those are (check git log -p for recent commits if helpful, and sgl-kernel / python/sglang/jit_kernel). Explain: (1) where layerwise offloading lives and how the whole-layer baseline ("Layerwise") works; (2) how chunked prefetching is implemented (chunk size config, pause/resume mechanism, how collectives signal the pause flag, CUDA events); (3) how partial residency is configured; (4) ALL relevant CLI flags / env vars / config knobs to switch between No-Offload, Layerwise (SGLang baseline), and ChunkFlow modes, including chunk size and residency fraction. Quote exact flag/env names and defaults. Report file paths with line numbers.`,
  },
  {
    key: 'harness',
    prompt: `${CTX}

Your task: understand the EXPERIMENT HARNESS. Read /workspace/sglang/offload_profiling/ (all run*.sh scripts, model_profile_common.sh, summarize/analyze scripts, trace_analysis dir). Explain: (1) exactly how the paper's experiments are launched (command lines, which sglang entrypoint / CLI, which models and model paths, frame sizes, SP degree, num denoising steps, prompts); (2) what each run*.sh variant does (run.sh vs run_no.sh vs run_old.sh vs run_access*.sh, profile variants); (3) how step time and peak memory are measured/extracted (which logs, which summarize script, what metrics); (4) what env vars control modes (offload on/off, chunk size, residency, etc.) as actually used by the scripts; (5) what would need to change to run on this 4-GPU box at SP degree 2 and 4 (GPU selection, model paths, expected download locations). Quote exact commands and env var names. Report file paths with line numbers.`,
  },
  {
    key: 'docs-history',
    prompt: `${CTX}

Your task: understand the PROJECT STATE from docs and git history. (1) Read any ChunkFlow/offload-related docs in the repo (search docs/, root *.md, python/sglang/**/README*, anything added by recent commits — run: git log --oneline -40, and git diff --stat against upstream base if identifiable; the branch is v0.5.9-shard). (2) Summarize the last ~40 commits: what was built in what order, what looks finished vs in-progress (e.g. "comm-window-mode kernel", "monitor work complete marker", "fix mem" commits). (3) Identify how this fork diverges from upstream sglang v0.5.9 at a high level (git diff --stat vs the merge-base with main or a version tag if one exists — try git tag -l and git merge-base). (4) Note anything about how the paper's numbers were produced (configs, hardware assumptions like H100, hard-coded constants like PCIe bandwidth or H100-specific tuning that might break on RTX 5090 / L40). Report file paths and commit hashes.`,
  },
  {
    key: 'multigpu',
    prompt: `${CTX}

Your task: assess MULTI-GPU (SP degree 4) and NEW-HARDWARE readiness of this fork. (1) Find how Ulysses sequence parallelism degree is configured for the diffusion/multimodal path (CLI flags like tp-size/sp-size/ulysses-degree, torchrun vs internal launcher, NCCL setup) and whether degree 4 is supported by the code (look for asserts, head-count divisibility for Wan2.2-TI2V-5B which the paper says has d=3072; check num attention heads of the Wan 5B model in the model code, and whether 4-way head split works). (2) Check for hardware-specific assumptions that could break on RTX 5090 (sm_120 Blackwell): custom CUDA kernels in sgl-kernel or python/sglang/jit_kernel used by the offload path (do they compile for sm_120? check setup.py / CMake arch lists / JIT arch detection), FlashAttention backend availability for Blackwell in the diffusion path (which attention backend does the multimodal-gen path use, what are the options). (3) Check python/pyproject dependencies: which torch version does this fork pin, and does the install path work with cu128+ (needed for sm_120). (4) Find any hard-coded 2-GPU assumptions in the offload/comm-window code (e.g. rank checks, PCIe BW constants, CUDA_VISIBLE_DEVICES). Report file paths with line numbers.`,
  },
]

phase('Read')
const results = await parallel(TASKS.map(t => () =>
  agent(t.prompt, { label: `read:${t.key}`, phase: 'Read', schema: SCHEMA })
))

return Object.fromEntries(TASKS.map((t, i) => [t.key, results[i]]))