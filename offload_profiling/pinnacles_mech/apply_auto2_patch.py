#!/usr/bin/env python3
"""auto mode, second iteration: better slack estimate.

* bandwidth: estimate from chunks copied while no collective was active (the free
  link), not from chunks that shared the link with NCCL;
* available time: (block time left) - (expected comm still to come in this block),
  using the comm time already observed in the block, instead of a uniform comm fraction;
* default margin 10%; a few decision samples are logged for inspection.
usage: apply_auto2_patch.py <comm_aware_prefetch.py> <layerwise_offload_chunkwise.py>
"""
import pathlib
import sys

tracker_path = pathlib.Path(sys.argv[1])
manager_path = pathlib.Path(sys.argv[2])

src = tracker_path.read_text()
if "remaining_comm" not in src:
    edits = [
        ('''        comm = self._block_comm.get(id(mgr), 0.0)
        elapsed = time.perf_counter() - t0
        avail = max(0.0, dur - elapsed) * (1.0 - min(0.9, comm / dur))
''', '''        comm = self._block_comm.get(id(mgr), 0.0)
        elapsed = time.perf_counter() - t0
        remaining_comm = max(0.0, comm - self._block_comm_acc)
        avail = max(0.0, (dur - elapsed) - remaining_comm)
'''),
        ('''        if bw <= 0:
            bw = 20e9
        return pending / bw <= avail * (1.0 - self._yield_margin)
''', '''        if bw <= 0:
            bw = 20e9
        decision = pending / bw <= avail * (1.0 - self._yield_margin)
        if self._auto_debug_left > 0:
            self._auto_debug_left -= 1
            logger.info(
                "[comm-auto] layer=%s elapsed=%.1fms dur=%.1fms comm=%.1fms comm_seen=%.1fms "
                "avail=%.1fms pending=%.0fMB bw=%.1fGB/s need=%.1fms -> %s",
                _layer, elapsed * 1e3, dur * 1e3, comm * 1e3, self._block_comm_acc * 1e3,
                avail * 1e3, pending / 1e6, bw / 1e9, pending / bw * 1e3,
                "yield" if decision else "share",
            )
        return decision
'''),
        ('''        self._yield_margin = float(os.getenv("SGLANG_DIT_COMM_YIELD_MARGIN", "0.15"))
''', '''        self._yield_margin = float(os.getenv("SGLANG_DIT_COMM_YIELD_MARGIN", "0.10"))
        self._auto_debug_left = int(os.getenv("SGLANG_DIT_COMM_AUTO_DEBUG", "60"))
'''),
        ('''import contextlib
import os
import queue
''', '''import contextlib
import logging
import os
import queue
'''),
        ('''class _KernelCommRegion:
''', '''logger = logging.getLogger(__name__)


class _KernelCommRegion:
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
if "_h2d_bw_ema_free" not in src:
    edits = [
        ('''        self._h2d_bw_ema = 0.0
''', '''        self._h2d_bw_ema = 0.0
        self._h2d_bw_ema_free = 0.0
'''),
        ('''    def h2d_bandwidth_estimate(self) -> float:
        return float(self._h2d_bw_ema)
''', '''    def h2d_bandwidth_estimate(self) -> float:
        # Prefer the bandwidth seen while no collective was active (the free link).
        return float(self._h2d_bw_ema_free or self._h2d_bw_ema)
'''),
        ('''            if self.comm_tracker is not None and self.comm_tracker.is_active():
                return False

            push_nvtx = bool(torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"))
''', '''            if self.comm_tracker is not None and self.comm_tracker.is_active():
                return False
            free_link = self.comm_tracker is None or not self.comm_tracker.is_active()

            push_nvtx = bool(torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"))
'''),
        ('''                self._h2d_bw_ema = (
                    inst_bw if self._h2d_bw_ema <= 0 else 0.9 * self._h2d_bw_ema + 0.1 * inst_bw
                )
''', '''                self._h2d_bw_ema = (
                    inst_bw if self._h2d_bw_ema <= 0 else 0.9 * self._h2d_bw_ema + 0.1 * inst_bw
                )
                if free_link and (self.comm_tracker is None or not self.comm_tracker.is_active()):
                    self._h2d_bw_ema_free = (
                        inst_bw if self._h2d_bw_ema_free <= 0
                        else 0.9 * self._h2d_bw_ema_free + 0.1 * inst_bw
                    )
'''),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"manager pattern not found exactly once:\n{old}"
        src = src.replace(old, new)
    manager_path.write_text(src)
    print("patched:", manager_path)
else:
    print("already patched:", manager_path)
