#!/usr/bin/env python3
"""auto mode, third iteration: one decision per forward, identical on every rank.

Per-collective decisions diverged between the two ranks (one yields while the other
shares, so the collective is slowed anyway and the yielding rank falls behind). Now the
tracker closes a "step" at every start marker of layer 0 of the first manager, measures
compute-only time = step duration - comm time, T_pref = prefetched bytes / free-link
bandwidth, and sets the mode for the next forward: yield iff T_pref <= T_comp*(1-margin),
with hysteresis. This is the paper's F* condition applied online.
usage: apply_auto3_patch.py <comm_aware_prefetch.py> <layerwise_offload_chunkwise.py>
"""
import pathlib
import sys

tracker_path = pathlib.Path(sys.argv[1])
manager_path = pathlib.Path(sys.argv[2])

src = tracker_path.read_text()
if "_close_step" not in src:
    # 1. replace _should_yield_now wholesale with the per-step mode lookup + _close_step
    a = src.index("    def _should_yield_now(self) -> bool:")
    b = src.index("\n    def ", a + 10) + 1
    new_fn = '''    def _should_yield_now(self) -> bool:
        """auto mode: the per-forward mode set by _close_step (identical on all ranks)."""
        if self._window_mode != "auto":
            return True
        return self._mode_yield

    def _close_step(self, now: float) -> None:
        """Called at the GPU start of layer 0 (first manager): close the previous forward,
        apply the online F* test and set the yield/share mode for the next forward."""
        t0 = self._step_t0
        comm = self._step_comm
        self._step_t0 = now
        self._step_comm = 0.0
        if t0 is None:
            return
        dur = now - t0
        nbytes = 0
        bw = 0.0
        for m in self._managers:
            try:
                nbytes += int(m.step_prefetch_bytes())
                bw = max(bw, float(m.h2d_bandwidth_estimate() or 0.0))
            except Exception:
                continue
        if bw <= 0:
            bw = 20e9
        need = nbytes / bw
        compute_only = max(0.0, dur - comm)
        ratio = need / compute_only if compute_only > 0 else float("inf")
        hi = 1.0 - self._yield_margin
        if self._mode_yield and ratio > hi + 0.05:
            self._mode_yield = False
        elif not self._mode_yield and ratio <= hi:
            self._mode_yield = True
        self._auto_steps += 1
        logger.info(
            "[comm-auto] forward %d: dur=%.0fms comm=%.0fms compute_only=%.0fms prefetch=%.0fMB "
            "bw=%.1fGB/s need=%.0fms ratio=%.2f -> %s",
            self._auto_steps, dur * 1e3, comm * 1e3, compute_only * 1e3, nbytes / 1e6,
            bw / 1e9, need * 1e3, ratio, "yield" if self._mode_yield else "share",
        )

'''
    src = src[:a] + new_fn + src[b:]
    edits = [
        ('''                    if kind == "start":
                        self._cur_block = (mgr, item["layer"], now)
                        self._block_comm_acc = 0.0
''', '''                    if kind == "start":
                        if item["layer"] == 0 and self._managers and mgr is self._managers[0]:
                            self._close_step(now)
                        self._cur_block = (mgr, item["layer"], now)
                        self._block_comm_acc = 0.0
'''),
        ('''                    self._block_comm_acc += time.perf_counter() - item["t0"]
''', '''                    region_dur = time.perf_counter() - item["t0"]
                    self._block_comm_acc += region_dur
                    self._step_comm += region_dur
'''),
        ('''        self._auto_stats: Dict[str, int] = {"yield": 0, "share": 0}
''', '''        self._auto_stats: Dict[str, int] = {"yield": 0, "share": 0}
        self._mode_yield = False
        self._step_t0: float | None = None
        self._step_comm = 0.0
        self._auto_steps = 0
'''),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"tracker pattern not found exactly once:\n{old}"
        src = src.replace(old, new)
    tracker_path.write_text(src)
    print("patched:", tracker_path)
else:
    print("already patched:", tracker_path)

src = manager_path.read_text()
if "step_prefetch_bytes" not in src:
    old = '''    def pending_prefetch_bytes(self) -> int:
'''
    new = '''    def step_prefetch_bytes(self) -> int:
        """Bytes prefetched per forward: every layer except layer 0, which stays resident."""
        return int(sum(b for l, b in self._layer_total_bytes.items() if l != 0))

    def pending_prefetch_bytes(self) -> int:
'''
    assert src.count(old) == 1
    src = src.replace(old, new)
    manager_path.write_text(src)
    print("patched:", manager_path)
else:
    print("already patched:", manager_path)
