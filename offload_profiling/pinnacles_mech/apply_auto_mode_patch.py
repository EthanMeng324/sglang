#!/usr/bin/env python3
"""Add comm window mode "auto": slack-aware yielding on top of the GPU-timed window.

When the compute stream reaches a collective, the tracker decides per collective:
yield (block chunk launches until the NCCL Work completes) if the bytes still to be
prefetched can be copied within the compute-only time left in the current block,
otherwise keep sharing the link. Block boundaries are learned from CUDA events the
offload manager records at every block's pre/post hook; the manager also reports
pending bytes and an EMA of the measured chunk copy bandwidth.
usage: apply_auto_mode_patch.py <comm_aware_prefetch.py> <layerwise_offload_chunkwise.py>
"""
import pathlib
import sys

tracker_path = pathlib.Path(sys.argv[1])
manager_path = pathlib.Path(sys.argv[2])

# ---------------------------------------------------------------- tracker
src = tracker_path.read_text()
if "_should_yield_now" not in src:
    old_loop_start = src.index("    def _enqueue_gpu_region(")
    old_loop_end = src.index("                time.sleep(0.0005)\n", old_loop_start) + len("                time.sleep(0.0005)\n")
    new_block = '''    def register_manager(self, manager: Any) -> None:
        if manager not in self._managers:
            self._managers.append(manager)

    def wants_block_markers(self) -> bool:
        return self._window_mode == "auto" and self._gpu_q is not None

    def note_block_marker(
        self, manager: Any, layer_idx: int, kind: str, event: torch.cuda.Event
    ) -> None:
        if self._gpu_q is None:
            return
        self._gpu_q.put({"kind": kind, "event": event, "manager": manager, "layer": layer_idx})

    def auto_stats(self) -> Dict[str, int]:
        return dict(self._auto_stats)

    def _enqueue_gpu_region(
        self,
        tag: str,
        start_event: torch.cuda.Event | None,
        *,
        nccl_work: Any = None,
        end_event: torch.cuda.Event | None = None,
    ) -> None:
        if self._gpu_q is None or start_event is None:
            return
        self._gpu_q.put(
            {"kind": "region", "state": 0, "start": start_event, "work": nccl_work,
             "end": end_event, "tag": tag, "t0": 0.0}
        )

    def _should_yield_now(self) -> bool:
        """auto mode: yield only if the pending prefetch fits in the compute-only
        time left in the current block (learned from the previous blocks)."""
        if self._window_mode != "auto":
            return True
        cur = self._cur_block
        if cur is None:
            return False
        mgr, _layer, t0 = cur
        dur = self._block_dur.get(id(mgr))
        if not dur:
            return False
        comm = self._block_comm.get(id(mgr), 0.0)
        elapsed = time.perf_counter() - t0
        avail = max(0.0, dur - elapsed) * (1.0 - min(0.9, comm / dur))
        pending = 0
        bw = 0.0
        for m in self._managers:
            try:
                pending += int(m.pending_prefetch_bytes())
                bw = max(bw, float(m.h2d_bandwidth_estimate() or 0.0))
            except Exception:
                continue
        if pending <= 0:
            return True
        if bw <= 0:
            bw = 20e9
        return pending / bw <= avail * (1.0 - self._yield_margin)

    def _gpu_region_loop(self) -> None:
        """Open each region when its start event completes on the GPU (in auto mode
        only if the prefetch has slack) and close it when its NCCL Work (or end
        event) completes; block markers teach the per-block duration and comm time."""
        assert self._gpu_q is not None
        assert self._event_stop_event is not None
        pending: list[dict] = []
        while True:
            while True:
                try:
                    item = self._gpu_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    for it in pending:
                        if it.get("state") == 1:
                            self._block_launches_end()
                    pending.clear()
                    return
                pending.append(item)

            progressed = False
            still: list[dict] = []
            for item in pending:
                kind = item["kind"]
                if kind != "region":
                    try:
                        done = item["event"].query()
                    except Exception:
                        done = True
                    if not done:
                        still.append(item)
                        continue
                    progressed = True
                    now = time.perf_counter()
                    mgr = item["manager"]
                    if kind == "start":
                        self._cur_block = (mgr, item["layer"], now)
                        self._block_comm_acc = 0.0
                    else:
                        cur = self._cur_block
                        if cur is not None and cur[0] is mgr:
                            d = now - cur[2]
                            key = id(mgr)
                            prev = self._block_dur.get(key)
                            self._block_dur[key] = d if prev is None else 0.7 * prev + 0.3 * d
                            prev_c = self._block_comm.get(key)
                            self._block_comm[key] = (
                                self._block_comm_acc if prev_c is None
                                else 0.7 * prev_c + 0.3 * self._block_comm_acc
                            )
                            self._cur_block = None
                    continue
                if item["state"] == 0:
                    try:
                        reached = item["start"].query()
                    except Exception:
                        reached = True
                    if not reached:
                        still.append(item)
                        continue
                    item["t0"] = time.perf_counter()
                    if self._should_yield_now():
                        self._block_launches_start()
                        item["state"] = 1
                        self._auto_stats["yield"] += 1
                    else:
                        item["state"] = 2
                        self._auto_stats["share"] += 1
                    progressed = True
                work, end_event = item["work"], item["end"]
                try:
                    if work is not None:
                        done = work.is_completed()
                    elif end_event is not None:
                        done = end_event.query()
                    else:
                        done = True
                except Exception:
                    done = True
                if done:
                    if item["state"] == 1:
                        self._block_launches_end()
                    self._block_comm_acc += time.perf_counter() - item["t0"]
                    progressed = True
                else:
                    still.append(item)
            pending = still

            if self._event_stop_event.is_set() and not pending:
                break
            if progressed:
                continue
            if not pending:
                try:
                    item = self._gpu_q.get(timeout=0.05)
                except queue.Empty:
                    if self._event_stop_event.is_set():
                        break
                    continue
                if item is None:
                    return
                pending.append(item)
            else:
                time.sleep(0.0005)
'''
    src = src[:old_loop_start] + new_block + src[old_loop_end:]
    edits = [
        ('''    if normalized in {"gpu", "device", "gpu_timed"}:
        return "gpu"
    return "kernel"
''', '''    if normalized in {"gpu", "device", "gpu_timed"}:
        return "gpu"
    if normalized in {"auto", "adaptive", "slack"}:
        return "auto"
    return "kernel"
'''),
        ('''        self._gpu_q: queue.Queue | None = None
        self._gpu_thread: threading.Thread | None = None
        if self._window_mode == "gpu" and torch.cuda.is_available():
''', '''        self._gpu_q: queue.Queue | None = None
        self._gpu_thread: threading.Thread | None = None
        self._managers: list[Any] = []
        self._cur_block: tuple[Any, int, float] | None = None
        self._block_dur: Dict[int, float] = {}
        self._block_comm: Dict[int, float] = {}
        self._block_comm_acc = 0.0
        self._yield_margin = float(os.getenv("SGLANG_DIT_COMM_YIELD_MARGIN", "0.15"))
        self._auto_stats: Dict[str, int] = {"yield": 0, "share": 0}
        if self._window_mode in ("gpu", "auto") and torch.cuda.is_available():
'''),
        ('''        return self._window_mode in ("kernel", "gpu")
''', '''        return self._window_mode in ("kernel", "gpu", "auto")
'''),
        ('''            self, tag or "comm", gpu_timed=(self._window_mode == "gpu")
''', '''            self, tag or "comm", gpu_timed=(self._window_mode in ("gpu", "auto"))
'''),
        ('''import contextlib
import queue
''', '''import contextlib
import os
import queue
'''),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"tracker pattern not found exactly once:\n{old}"
        src = src.replace(old, new)
    tracker_path.write_text(src)
    print("patched:", tracker_path)
else:
    print("already patched:", tracker_path)

# ---------------------------------------------------------------- manager
src = manager_path.read_text()
if "pending_prefetch_bytes" not in src:
    edits = [
        ('''        self._layer_name_re = re.compile(
''', '''        if self.comm_tracker is not None:
            register = getattr(self.comm_tracker, "register_manager", None)
            if callable(register):
                register(self)
        self._h2d_bw_ema = 0.0

        self._layer_name_re = re.compile(
'''),
        ('''            self._record_copy_stats(background=background, nbytes=nbytes, dur_s=copy_dur_s)
''', '''            self._record_copy_stats(background=background, nbytes=nbytes, dur_s=copy_dur_s)
            if copy_dur_s > 0:
                inst_bw = nbytes / copy_dur_s
                self._h2d_bw_ema = (
                    inst_bw if self._h2d_bw_ema <= 0 else 0.9 * self._h2d_bw_ema + 0.1 * inst_bw
                )
'''),
        ('''    def _prefetch_worker_loop(self) -> None:
''', '''    def h2d_bandwidth_estimate(self) -> float:
        return float(self._h2d_bw_ema)

    def pending_prefetch_bytes(self) -> int:
        """Bytes still to be copied for every layer currently scheduled for prefetch."""
        with self._state_lock:
            total = 0
            for layer_idx in self._scheduled_layers:
                remaining = int(self._layer_total_bytes.get(layer_idx, 0))
                cpu_phases = self._consolidated_cpu_weights.get(layer_idx, {})
                for phase_idx, by_dtype in self._prefetch_offsets.get(layer_idx, {}).items():
                    for dtype, numel in by_dtype.items():
                        buf = cpu_phases.get(phase_idx, {}).get(dtype)
                        if buf is not None:
                            remaining -= int(numel) * buf.element_size()
                total += max(0, remaining)
            return total

    @torch.compiler.disable
    def _note_block_marker(self, kind: str, layer_idx: int) -> None:
        tracker = self.comm_tracker
        if tracker is None:
            return
        wants = getattr(tracker, "wants_block_markers", None)
        if not callable(wants) or not wants():
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        tracker.note_block_marker(self, layer_idx, kind, event)

    def _prefetch_worker_loop(self) -> None:
'''),
        ('''                self._ensure_phase_ready(i, target_phase_idx=entry_target_phase_idx)
                self._record_prefetch_cp_wait_end(step_idx, i, wait_start)
''', '''                self._ensure_phase_ready(i, target_phase_idx=entry_target_phase_idx)
                self._record_prefetch_cp_wait_end(step_idx, i, wait_start)
                self._note_block_marker("start", i)
'''),
        ('''            def hook(module, input, output):
                self.release_layer(i)
''', '''            def hook(module, input, output):
                self._note_block_marker("end", i)
                self.release_layer(i)
'''),
        ('''            f"copy_dup_ratio={dup_ratio:.3f}"
        )
''', '''            f"copy_dup_ratio={dup_ratio:.3f} "
            f"auto={_auto_stats_str(self.comm_tracker)}"
        )
'''),
        ('''def _parse_phase_name_csv(raw: str | None) -> Set[str]:
''', '''def _auto_stats_str(tracker) -> str:
    fn = getattr(tracker, "auto_stats", None)
    if not callable(fn):
        return "-"
    try:
        st = fn()
        return f"yield={st.get('yield', 0)}/share={st.get('share', 0)}"
    except Exception:
        return "-"


def _parse_phase_name_csv(raw: str | None) -> Set[str]:
'''),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"manager pattern not found exactly once:\n{old}"
        src = src.replace(old, new)
    manager_path.write_text(src)
    print("patched:", manager_path)
else:
    print("already patched:", manager_path)
