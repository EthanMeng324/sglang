import contextlib
import ctypes
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict

import torch


def _normalize_window_mode(mode: str | None) -> str:
    normalized = (mode or "kernel").strip().lower()
    if normalized in {"kernel", "event", "stream_event"}:
        return "kernel"
    if normalized in {"launch", "host", "wrapper"}:
        return "launch"
    if normalized in {"none", "off", "noyield"}:
        return "none"
    if normalized in {"gpu", "device", "gpu_timed"}:
        return "gpu"
    if normalized in {"auto", "adaptive", "slack"}:
        return "auto"
    if normalized in {"gate", "memops", "stream_gate"}:
        return "gate"
    return "kernel"


logger = logging.getLogger(__name__)


def _find_libcuda() -> str:
    try:
        with open("/proc/self/maps") as f:
            for ln in f:
                if "libcuda.so" in ln:
                    return ln.split()[-1]
    except Exception:
        pass
    return "libcuda.so.1"


class _StreamGate:
    """GPU-side prefetch gate on CUDA stream memory operations.

    The compute stream writes a device flag to 1 right before each collective and
    back to 0 right after it (stream-ordered: the writes land when the GPU reaches
    those points). The copy stream waits for flag == 0 before every chunk, so a
    chunk already in flight when the flag flips is the only traffic that overlaps
    the collective. No host thread is involved in opening or closing the window.
    """

    CU_STREAM_WAIT_VALUE_EQ = 0x1

    def __init__(self, device: torch.device) -> None:
        self.device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        lib = ctypes.CDLL(_find_libcuda())
        u32 = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self._write = self._fn(lib, ("cuStreamWriteValue32_v2", "cuStreamWriteValue32"), u32)
        self._wait = self._fn(lib, ("cuStreamWaitValue32_v2", "cuStreamWaitValue32"), u32)
        self._ctx_get = self._fn(lib, ("cuCtxGetCurrent",), [ctypes.POINTER(ctypes.c_void_p)])
        self._ctx_set = self._fn(lib, ("cuCtxSetCurrent",), [ctypes.c_void_p])
        self._primary_retain = self._fn(
            lib, ("cuDevicePrimaryCtxRetain",), [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        )
        self._lib = lib
        self._primary_ctx: int | None = None
        # the flag lives in a small device tensor kept alive by this object
        self._flag = torch.zeros(
            64, dtype=torch.int32, device=torch.device("cuda", self.device_index)
        )
        self.ptr = int(self._flag.data_ptr())
        self.n_writes = 0
        self.n_waits = 0
        self._ensure_ctx()
        rc = self._write(
            ctypes.c_void_p(torch.cuda.current_stream(self.device_index).cuda_stream),
            ctypes.c_uint64(self.ptr), ctypes.c_uint32(0), ctypes.c_uint(0),
        )
        if rc != 0:
            raise RuntimeError(f"cuStreamWriteValue32 probe failed (rc={rc})")

    @staticmethod
    def _fn(lib, names, argtypes):
        for n in names:
            f = getattr(lib, n, None)
            if f is not None:
                f.restype = ctypes.c_int
                f.argtypes = argtypes
                return f
        raise RuntimeError(f"none of {names} found in libcuda")

    def _ensure_ctx(self) -> None:
        # the driver API needs a current context in the calling thread; torch's
        # primary context may not be current yet in a freshly started worker thread
        ctx = ctypes.c_void_p(0)
        self._ctx_get(ctypes.byref(ctx))
        if ctx.value:
            return
        if self._primary_ctx is None:
            p = ctypes.c_void_p(0)
            rc = self._primary_retain(ctypes.byref(p), ctypes.c_int(self.device_index))
            if rc != 0:
                raise RuntimeError(f"cuDevicePrimaryCtxRetain rc={rc}")
            self._primary_ctx = p.value
        self._ctx_set(ctypes.c_void_p(self._primary_ctx))

    def write(self, stream: torch.cuda.Stream, value: int) -> None:
        self._ensure_ctx()
        rc = self._write(
            ctypes.c_void_p(stream.cuda_stream), ctypes.c_uint64(self.ptr),
            ctypes.c_uint32(value), ctypes.c_uint(0),
        )
        if rc != 0:
            raise RuntimeError(f"cuStreamWriteValue32 rc={rc}")
        self.n_writes += 1

    def wait_zero(self, stream: torch.cuda.Stream) -> None:
        self._ensure_ctx()
        rc = self._wait(
            ctypes.c_void_p(stream.cuda_stream), ctypes.c_uint64(self.ptr),
            ctypes.c_uint32(0), ctypes.c_uint(self.CU_STREAM_WAIT_VALUE_EQ),
        )
        if rc != 0:
            raise RuntimeError(f"cuStreamWaitValue32 rc={rc}")
        self.n_waits += 1


class _KernelCommRegion:
    def __init__(
        self,
        tracker: "CommunicationActivityTracker",
        tag: str,
        gpu_timed: bool = False,
        gate: Any = None,
    ):
        self._tracker = tracker
        self._tag = tag
        self._end_event = torch.cuda.Event()
        self._finalized = False
        self._active_started = False
        self._nccl_work: Any = None
        self._gpu_timed = gpu_timed
        self._start_event: torch.cuda.Event | None = None
        self._gate = gate

    def record_start(self) -> None:
        if self._active_started:
            return
        if self._gate is not None:
            # Stream-ordered gate: flag=1 lands when the compute stream reaches the
            # collective; the copy stream stops issuing chunks from that point.
            self._gate.write(torch.cuda.current_stream(), 1)
            self._active_started = True
            return
        if self._gpu_timed:
            # GPU-timed window: the region becomes active only when the compute
            # stream reaches this point on the GPU, not when the host enqueues
            # the collective (the host runs a queue depth ahead of the GPU).
            self._start_event = torch.cuda.Event()
            self._start_event.record(torch.cuda.current_stream())
            self._active_started = True
            return
        self._tracker._block_launches_start()
        self._active_started = True

    def set_nccl_work(self, work: Any) -> None:
        """Attach an NCCL Work handle so the active window extends until the
        collective truly completes on the GPU, not just until the host-side
        launch returns."""
        self._nccl_work = work

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if not self._active_started:
            return
        if self._gate is not None:
            # The current stream already waits on the collective's Work, so this
            # write lands right after the collective completes on the GPU.
            self._gate.write(torch.cuda.current_stream(), 0)
            return
        if self._gpu_timed:
            end_event = None
            if self._nccl_work is None:
                end_event = torch.cuda.Event()
                end_event.record(torch.cuda.current_stream())
            self._tracker._enqueue_gpu_region(
                self._tag,
                self._start_event,
                nccl_work=self._nccl_work,
                end_event=end_event,
            )
            return
        if self._nccl_work is not None:
            self._tracker._enqueue_kernel_region(
                self._tag, None, nccl_work=self._nccl_work
            )
        else:
            self._end_event.record(torch.cuda.current_stream())
            self._tracker._enqueue_kernel_region(self._tag, self._end_event)


class CommunicationActivityTracker:
    """Track whether communication is currently active.

    The tracker is framework-agnostic and only exposes a boolean activity signal.
    You can drive it manually with `comm_region()`, or patch torch.distributed.
    """

    def __init__(self, window_mode: str = "kernel") -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._active = 0
        self._patch_state: Dict[str, Any] | None = None
        self._window_mode = _normalize_window_mode(window_mode)
        self._event_q: queue.Queue[tuple[torch.cuda.Event, str]] | None = None
        self._event_stop_event: threading.Event | None = None
        self._event_thread: threading.Thread | None = None
        self._work_q: queue.Queue[tuple[Any, str] | None] | None = None
        self._work_thread: threading.Thread | None = None
        if self._window_mode == "kernel" and torch.cuda.is_available():
            self._event_q = queue.Queue()
            self._work_q = queue.Queue()
            self._event_stop_event = threading.Event()
            self._event_thread = threading.Thread(
                target=self._kernel_event_loop,
                name="CommActivityTrackerKernelEvents",
                daemon=True,
            )
            self._event_thread.start()
            self._work_thread = threading.Thread(
                target=self._nccl_work_wait_loop,
                name="CommActivityTrackerWorkWaiter",
                daemon=True,
            )
            self._work_thread.start()
        self._gpu_q: queue.Queue | None = None
        self._gpu_thread: threading.Thread | None = None
        self._managers: list[Any] = []
        self._gate: _StreamGate | None = None
        self._gate_failed = False
        self._cur_block: tuple[Any, int, float] | None = None
        self._block_dur: Dict[int, float] = {}
        self._block_comm: Dict[int, float] = {}
        self._block_comm_acc = 0.0
        self._yield_margin = float(os.getenv("SGLANG_DIT_COMM_YIELD_MARGIN", "0.10"))
        self._auto_debug_left = int(os.getenv("SGLANG_DIT_COMM_AUTO_DEBUG", "60"))
        self._auto_stats: Dict[str, int] = {"yield": 0, "share": 0}
        self._phase = "warm"
        self._dur_share: float | None = None
        self._dur_yield: float | None = None
        self._next_check = 0
        self._period = 16
        self._yield_tolerance = float(os.getenv("SGLANG_DIT_COMM_YIELD_TOLERANCE", "0.01"))
        self._step_t0: float | None = None
        self._step_comm = 0.0
        self._auto_steps = 0
        if self._window_mode in ("gpu", "auto") and torch.cuda.is_available():
            self._gpu_q = queue.Queue()
            if self._event_stop_event is None:
                self._event_stop_event = threading.Event()
            self._gpu_thread = threading.Thread(
                target=self._gpu_region_loop,
                name="CommActivityTrackerGpuRegions",
                daemon=True,
            )
            self._gpu_thread.start()

    @property
    def window_mode(self) -> str:
        return self._window_mode

    def stream_gate(self, device: torch.device | None = None):
        """The GPU-side gate (window mode "gate"), created on first use; None otherwise."""
        if self._window_mode != "gate":
            return None
        with self._lock:
            if self._gate is None and not self._gate_failed:
                try:
                    dev = device if device is not None else torch.device(
                        "cuda", torch.cuda.current_device()
                    )
                    self._gate = _StreamGate(dev)
                    logger.info(
                        "[comm-gate] stream memops gate ready on %s (flag at 0x%x)",
                        dev, self._gate.ptr,
                    )
                except Exception as ex:
                    self._gate_failed = True
                    logger.warning(
                        "[comm-gate] stream memops gate unavailable (%r); "
                        "collectives fall back to host-side launch windows", ex,
                    )
            return self._gate

    def gate_stats(self) -> Dict[str, int]:
        g = self._gate
        return {"writes": g.n_writes, "waits": g.n_waits} if g is not None else {}

    def uses_kernel_window(self) -> bool:
        # These modes launch the collective with async_op=True and track its Work.
        return self._window_mode in ("kernel", "gpu", "auto", "gate")

    def begin_kernel_region(self, tag: str | None = None) -> _KernelCommRegion | None:
        if not self.uses_kernel_window() or not torch.cuda.is_available():
            return None
        if self._window_mode == "gate":
            gate = self.stream_gate()
            if gate is None:
                return None
            return _KernelCommRegion(self, tag or "comm", gate=gate)
        return _KernelCommRegion(
            self, tag or "comm", gpu_timed=(self._window_mode in ("gpu", "auto"))
        )

    def register_manager(self, manager: Any) -> None:
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
                        if item["layer"] == 0 and self._managers and mgr is self._managers[0]:
                            self._close_step(now)
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
                    region_dur = time.perf_counter() - item["t0"]
                    self._block_comm_acc += region_dur
                    self._step_comm += region_dur
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

    def _block_launches_start(self) -> None:
        if self._window_mode == "none":
            return
        with self._cv:
            self._active += 1
            self._cv.notify_all()

    def _block_launches_end(self) -> None:
        with self._cv:
            if self._active > 0:
                self._active -= 1
            self._cv.notify_all()

    def _enqueue_kernel_region(
        self,
        tag: str,
        end_event: torch.cuda.Event | None,
        *,
        nccl_work: Any = None,
    ) -> None:
        if nccl_work is not None and self._work_q is not None:
            self._work_q.put((nccl_work, tag))
            return
        if end_event is not None and self._event_q is not None:
            self._event_q.put((end_event, tag))
            return
        self._block_launches_end()

    def _nccl_work_wait_loop(self) -> None:
        """Poll NCCL Work objects for completion in parallel.

        Unlike a FIFO blocking approach (which can only retire ~4 Works/sec
        and causes unbounded queue growth), this loop polls ALL pending Work
        objects each iteration so completed ones are retired immediately.
        The pending list only holds truly in-flight Works (bounded by the
        NCCL pipeline depth, typically 5-15).
        """
        assert self._work_q is not None
        assert self._event_stop_event is not None
        pending: list[tuple[Any, str]] = []
        while True:
            # Drain all new items from the queue without blocking.
            while True:
                try:
                    item = self._work_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    for w, _ in pending:
                        try:
                            w.wait()
                        except Exception:
                            pass
                        self._block_launches_end()
                    pending.clear()
                    return
                pending.append(item)

            # Poll all pending Works for completion.
            made_progress = False
            next_pending: list[tuple[Any, str]] = []
            for work, tag in pending:
                try:
                    completed = work.is_completed()
                except Exception:
                    completed = True
                if completed:
                    del work
                    self._block_launches_end()
                    made_progress = True
                else:
                    next_pending.append((work, tag))
            pending = next_pending

            if self._event_stop_event.is_set() and not pending:
                break
            if made_progress:
                continue

            if not pending:
                try:
                    item = self._work_q.get(timeout=0.05)
                except queue.Empty:
                    if self._event_stop_event.is_set():
                        break
                    continue
                if item is None:
                    return
                pending.append(item)
            else:
                time.sleep(0.001)

    def _kernel_event_loop(self) -> None:
        assert self._event_q is not None
        assert self._event_stop_event is not None
        pending: list[tuple[torch.cuda.Event, str]] = []
        while True:
            made_progress = False
            while True:
                try:
                    item = self._event_q.get_nowait()
                except queue.Empty:
                    break
                pending.append(item)
                made_progress = True

            next_pending: list[tuple[torch.cuda.Event, str]] = []
            for end_event, tag in pending:
                if end_event.query():
                    self._block_launches_end()
                    made_progress = True
                    continue
                next_pending.append((end_event, tag))
            pending = next_pending

            if self._event_stop_event.is_set() and not pending:
                break
            if made_progress:
                continue
            try:
                item = self._event_q.get(timeout=0.001)
            except queue.Empty:
                continue
            pending.append(item)

    def mark_start(self, _tag: str | None = None) -> None:
        if self._window_mode == "none":
            return
        with self._cv:
            self._active += 1
            self._cv.notify_all()

    def mark_end(self, _tag: str | None = None) -> None:
        if self._window_mode == "none":
            return
        with self._cv:
            if self._active > 0:
                self._active -= 1
            self._cv.notify_all()

    def is_active(self) -> bool:
        with self._lock:
            return self._active > 0

    def wait_until_inactive(
        self,
        *,
        stop_event: threading.Event | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Wait until no communication is active.

        Returns:
            bool: True if tracker became inactive, False if timeout/stop_event happened first.
        """
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._cv:
            while self._active > 0:
                if stop_event is not None and stop_event.is_set():
                    return False
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(timeout=remaining)
                else:
                    self._cv.wait(timeout=0.05)
        return True

    @contextlib.contextmanager
    def comm_region(self, tag: str | None = None):
        self.mark_start(tag)
        try:
            yield
        finally:
            self.mark_end(tag)

    def patch_torch_distributed(self) -> bool:
        """Best-effort patch of selected torch.distributed APIs.

        Returns:
            bool: Whether patching succeeded.
        """
        if self._patch_state is not None:
            return True

        try:
            import torch.distributed as dist
        except Exception:
            return False

        if not dist.is_available():
            return False

        targets = [
            "all_to_all_single",
            "all_to_all",
            "all_reduce",
            "reduce_scatter_tensor",
            "all_gather_into_tensor",
            "send",
            "recv",
            "isend",
            "irecv",
            "batch_isend_irecv",
        ]

        originals: Dict[str, Callable[..., Any]] = {}

        class _WorkWrapper:
            def __init__(self, work: Any, tracker: "CommunicationActivityTracker"):
                self._work = work
                self._tracker = tracker
                self._done = False

            def _finish_once(self):
                if not self._done:
                    self._done = True
                    self._tracker.mark_end("dist_work_wait")

            def wait(self, *args, **kwargs):
                try:
                    return self._work.wait(*args, **kwargs)
                finally:
                    self._finish_once()

            def is_completed(self, *args, **kwargs):
                done = self._work.is_completed(*args, **kwargs)
                if done:
                    self._finish_once()
                return done

            def __getattr__(self, name):
                return getattr(self._work, name)

        def make_wrapper(fn: Callable[..., Any], name: str):
            def wrapped(*args, **kwargs):
                region = self.begin_kernel_region(name)
                if region is None:
                    self.mark_start(name)
                else:
                    region.record_start()
                try:
                    result = fn(*args, **kwargs)
                    async_op = bool(kwargs.get("async_op", False))
                    if region is not None:
                        region.finalize()
                        return result
                    if async_op and hasattr(result, "wait"):
                        return _WorkWrapper(result, self)
                    self.mark_end(name)
                    return result
                except Exception:
                    if region is not None:
                        region.finalize()
                    else:
                        self.mark_end(name)
                    raise

            return wrapped

        for name in targets:
            fn = getattr(dist, name, None)
            if callable(fn):
                originals[name] = fn
                setattr(dist, name, make_wrapper(fn, name))

        if not originals:
            return False

        self._patch_state = {"dist": dist, "originals": originals}
        return True

    def unpatch_torch_distributed(self) -> None:
        if self._patch_state is None:
            return
        dist = self._patch_state["dist"]
        originals = self._patch_state["originals"]
        for name, fn in originals.items():
            setattr(dist, name, fn)
        self._patch_state = None

    def close(self) -> None:
        if self._event_stop_event is not None:
            self._event_stop_event.set()
        if self._gpu_q is not None:
            self._gpu_q.put(None)
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=2.0)
            self._gpu_thread = None
        if self._work_q is not None:
            self._work_q.put(None)
        if self._work_thread is not None:
            self._work_thread.join(timeout=2.0)
            self._work_thread = None
        if self._event_thread is not None:
            self._event_thread.join(timeout=1.0)
            self._event_thread = None
