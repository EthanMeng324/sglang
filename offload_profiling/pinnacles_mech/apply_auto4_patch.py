#!/usr/bin/env python3
"""auto mode, fourth iteration: measure-and-compare mode selection.

warm (2 forwards, share) -> share-measure (online F* test on a stall-free forward) ->
yield trial (one forward) -> keep whichever forward was faster; re-check periodically
(period doubles after a failed trial, resets after a successful one). All quantities are
per-forward durations that both ranks observe identically.
usage: apply_auto4_patch.py <comm_aware_prefetch.py>
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
src = path.read_text()
if "_phase" in src and "yield trial" in src:
    print("already patched:", path)
    sys.exit(0)

a = src.index("    def _should_yield_now(self) -> bool:")
b = src.index("    def _gpu_region_loop(self) -> None:")
new_block = '''    def _should_yield_now(self) -> bool:
        """auto mode: yield in the trial/yield phases chosen by _close_step."""
        if self._window_mode != "auto":
            return True
        return self._phase in ("trial", "yield")

    def _close_step(self, now: float) -> None:
        """Called at the GPU start of layer 0 (first manager): close the previous forward
        and pick the mode for the next one. warm -> share (measure, online F* test) ->
        yield trial (one forward) -> keep the faster mode; re-check every `_period`
        forwards (period doubles after a failed trial)."""
        t0 = self._step_t0
        comm = self._step_comm
        self._step_t0 = now
        self._step_comm = 0.0
        if t0 is None:
            return
        dur = now - t0
        self._auto_steps += 1
        k = self._auto_steps
        phase = self._phase
        if k <= 2:
            self._phase = "share"
            self._auto_log(k, phase, dur, comm, None, None, "warm-up")
            return
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
        note = ""
        if phase == "share":
            self._dur_share = dur
            if k >= self._next_check:
                if ratio <= 1.0 - self._yield_margin:
                    self._phase = "trial"
                    note = "F* test passed -> yield trial"
                else:
                    self._next_check = k + self._period
                    note = "F* test failed -> stay share"
        elif phase == "trial":
            self._dur_yield = dur
            if self._dur_share is not None and dur <= self._dur_share * (1.0 - self._yield_tolerance):
                self._phase = "yield"
                self._period = 16
                self._next_check = k + self._period
                note = f"yield {dur*1e3:.0f}ms < share {self._dur_share*1e3:.0f}ms -> keep yield"
            else:
                self._phase = "share"
                self._period = min(self._period * 2, 256)
                self._next_check = k + self._period
                note = f"yield {dur*1e3:.0f}ms >= share {(self._dur_share or 0)*1e3:.0f}ms -> back to share"
        elif phase == "yield":
            self._dur_yield = dur
            if k >= self._next_check:
                self._phase = "share"
                note = "periodic re-check -> one share forward"
        self._auto_log(k, phase, dur, comm, ratio, need, note)

    def _auto_log(self, k, phase, dur, comm, ratio, need, note) -> None:
        logger.info(
            "[comm-auto] forward %d (%s): dur=%.0fms comm=%.0fms compute_only=%.0fms need=%s ratio=%s -> next=%s %s",
            k, phase, dur * 1e3, comm * 1e3, max(0.0, dur - comm) * 1e3,
            "-" if need is None else f"{need*1e3:.0f}ms", "-" if ratio is None else f"{ratio:.2f}",
            self._phase, note,
        )

'''
src = src[:a] + new_block + src[b:]
old = '''        self._mode_yield = False
        self._step_t0: float | None = None
'''
new = '''        self._phase = "warm"
        self._dur_share: float | None = None
        self._dur_yield: float | None = None
        self._next_check = 0
        self._period = 16
        self._yield_tolerance = float(os.getenv("SGLANG_DIT_COMM_YIELD_TOLERANCE", "0.01"))
        self._step_t0: float | None = None
'''
assert src.count(old) == 1, "init fields not found"
src = src.replace(old, new)
assert "_mode_yield" not in src, "stale _mode_yield reference remains"
path.write_text(src)
print("patched:", path)
