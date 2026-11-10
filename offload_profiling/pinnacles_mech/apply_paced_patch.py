#!/usr/bin/env python3
"""Rebuttal baseline (reviewer PShp W1): "submit prefetches at fixed intervals within the computation
interval". The prefetch worker issues the (synchronous) chunk copies on a fixed cadence instead of
back-to-back: SGLANG_DIT_PREFETCH_PACE_GBPS sets the average issue rate; the worker sleeps between chunks
so that chunk_bytes / rate elapses per chunk. No yielding to collectives (use window mode "none").
usage: apply_paced_patch.py [repo_root]"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
path = root / "python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py"
src = path.read_text()
if "_pace_next_t" in src:
    print("already patched:", path)
    sys.exit(0)
edits = [
    ('''        self._gate_inflight_cap = max(1, _env_int("SGLANG_DIT_GATE_INFLIGHT_CHUNKS", 64))
''',
     '''        self._gate_inflight_cap = max(1, _env_int("SGLANG_DIT_GATE_INFLIGHT_CHUNKS", 64))
        # fixed-cadence issue (rebuttal baseline): average chunk issue rate in GB/s, 0 = off
        self._pace_gbps = float(os.getenv("SGLANG_DIT_PREFETCH_PACE_GBPS", "0") or 0)
        self._pace_next_t = 0.0
'''),
    ('''            copied = self._copy_one_chunk(layer_idx, phase_idx, background=True)
            if not copied:
                time.sleep(0.0005)
''',
     '''            if self._pace_gbps > 0:
                now = time.perf_counter()
                if now < self._pace_next_t:
                    time.sleep(self._pace_next_t - now)
                    now = time.perf_counter()
            copied = self._copy_one_chunk(layer_idx, phase_idx, background=True)
            if copied and self._pace_gbps > 0:
                self._pace_next_t = max(now, self._pace_next_t) + (
                    self.prefetch_chunk_size_bytes / (self._pace_gbps * 1e9)
                )
            if not copied:
                time.sleep(0.0005)
'''),
]
for old, new in edits:
    assert src.count(old) == 1, f"pattern not found exactly once:\n{old}"
    src = src.replace(old, new)
path.write_text(src)
print("patched:", path)
