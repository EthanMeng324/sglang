#!/usr/bin/env python3
"""Add comm window mode "gate": a GPU-side prefetch gate built on CUDA stream memory operations.

Compute stream: flag=1 (cuStreamWriteValue32) right before each collective, flag=0 right after it
(the current stream already waits on the collective's Work). Copy stream: cuStreamWaitValue32(flag==0)
before every chunk, chunks issued asynchronously (non_blocking=True) by the prefetch worker with a
bounded enqueue-ahead depth. The per-collective host drain of the copy stream (quiesce_prefetch_for_comm)
is skipped in this mode. The only leak into a collective is the chunk already in flight when the flag flips.

usage: apply_gate_mode_patch.py [repo_root]
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
tracker_path = root / "python/sglang/multimodal_gen/runtime/utils/comm_aware_prefetch.py"
manager_path = root / "python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py"
harness_path = root / "offload_profiling/pinnacles_mech/run_mech.sh"


def apply(path: pathlib.Path, marker: str, edits):
    src = path.read_text()
    if marker in src:
        print("already patched:", path)
        return
    for old, new in edits:
        assert src.count(old) == 1, f"{path}: pattern not found exactly once:\n{old}"
        src = src.replace(old, new)
    path.write_text(src)
    print("patched:", path)


GATE_CLASS = '''logger = logging.getLogger(__name__)


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
'''

apply(tracker_path, "class _StreamGate", [
    ("import contextlib\nimport logging\nimport os\n",
     "import contextlib\nimport ctypes\nimport logging\nimport os\n"),
    ('''    if normalized in {"auto", "adaptive", "slack"}:
        return "auto"
    return "kernel"
''',
     '''    if normalized in {"auto", "adaptive", "slack"}:
        return "auto"
    if normalized in {"gate", "memops", "stream_gate"}:
        return "gate"
    return "kernel"
'''),
    ("logger = logging.getLogger(__name__)\n\n\nclass _KernelCommRegion:\n", GATE_CLASS),
    ('''        tag: str,
        gpu_timed: bool = False,
    ):
        self._tracker = tracker
''',
     '''        tag: str,
        gpu_timed: bool = False,
        gate: Any = None,
    ):
        self._tracker = tracker
'''),
    ('''        self._gpu_timed = gpu_timed
        self._start_event: torch.cuda.Event | None = None

    def record_start(self) -> None:
        if self._active_started:
            return
        if self._gpu_timed:
''',
     '''        self._gpu_timed = gpu_timed
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
'''),
    ('''        self._finalized = True
        if not self._active_started:
            return
        if self._gpu_timed:
            end_event = None
''',
     '''        self._finalized = True
        if not self._active_started:
            return
        if self._gate is not None:
            # The current stream already waits on the collective's Work, so this
            # write lands right after the collective completes on the GPU.
            self._gate.write(torch.cuda.current_stream(), 0)
            return
        if self._gpu_timed:
            end_event = None
'''),
    ("        self._managers: list[Any] = []\n",
     "        self._managers: list[Any] = []\n        self._gate: _StreamGate | None = None\n        self._gate_failed = False\n"),
    ('''    def uses_kernel_window(self) -> bool:
        # Both modes launch the collective with async_op=True and track its Work.
        return self._window_mode in ("kernel", "gpu", "auto")

    def begin_kernel_region(self, tag: str | None = None) -> _KernelCommRegion | None:
        if not self.uses_kernel_window() or not torch.cuda.is_available():
            return None
        return _KernelCommRegion(
            self, tag or "comm", gpu_timed=(self._window_mode in ("gpu", "auto"))
        )
''',
     '''    @property
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
'''),
])

apply(manager_path, "_gate_reap_inflight", [
    ("import os\nimport re\nimport threading\nimport time\nfrom contextlib import contextmanager\n",
     "import os\nimport re\nimport threading\nimport time\nfrom collections import deque\nfrom contextlib import contextmanager\n"),
    ('''        self.comm_active_window_mode = comm_active_window_mode
        self.prefetch_chunk_size_bytes = max(1, prefetch_chunk_size_mb) * 1024 * 1024
''',
     '''        self.comm_active_window_mode = comm_active_window_mode
        self.prefetch_chunk_size_bytes = max(1, prefetch_chunk_size_mb) * 1024 * 1024
        # window mode "gate": GPU-gated asynchronous chunk issue (see comm_aware_prefetch._StreamGate)
        self._gate_mode = str(comm_active_window_mode or "").strip().lower() in {
            "gate", "memops", "stream_gate"
        }
        self._stream_gate = None
        self._gate_inflight: deque = deque()
        self._gate_inflight_cap = max(1, _env_int("SGLANG_DIT_GATE_INFLIGHT_CHUNKS", 64))
'''),
    ('''    @torch.compiler.disable
    def _copy_one_chunk(
        self, layer_idx: int, phase_idx: int, *, background: bool
    ) -> bool:
''',
     '''    def _stream_gate_for_copy(self):
        if not self._gate_mode:
            return None
        if self._stream_gate is None and self.comm_tracker is not None:
            self._stream_gate = self.comm_tracker.stream_gate(self.device)
            if self._stream_gate is None:
                # gate unavailable: fall back to the blocking per-chunk copies
                self._gate_mode = False
        return self._stream_gate

    def _gate_reap_inflight(self, *, wait_oldest: bool = False) -> None:
        """Retire completed asynchronous chunks; their durations come from the events."""
        q = self._gate_inflight
        if wait_oldest and q:
            q[0][1].synchronize()
        while q and q[0][1].query():
            start_event, done_event, nbytes, background = q.popleft()
            try:
                dur_s = max(0.0, float(start_event.elapsed_time(done_event))) / 1e3
            except Exception:
                dur_s = 0.0
            self._record_copy_stats(background=background, nbytes=nbytes, dur_s=dur_s)
            if dur_s > 0:
                inst_bw = nbytes / dur_s
                self._h2d_bw_ema = (
                    inst_bw if self._h2d_bw_ema <= 0 else 0.9 * self._h2d_bw_ema + 0.1 * inst_bw
                )
                self._h2d_bw_ema_free = (
                    inst_bw if self._h2d_bw_ema_free <= 0
                    else 0.9 * self._h2d_bw_ema_free + 0.1 * inst_bw
                )

    def _gate_throttle_inflight(self) -> None:
        # The gate itself needs no bound; this keeps the enqueue-ahead depth (and the
        # event bookkeeping) small. 64 x 4 MB = one block of Wan 5B.
        while len(self._gate_inflight) >= self._gate_inflight_cap:
            self._gate_reap_inflight(wait_oldest=True)

    @torch.compiler.disable
    def _copy_one_chunk(
        self, layer_idx: int, phase_idx: int, *, background: bool
    ) -> bool:
'''),
    ('''            try:
                copy_t0 = time.perf_counter()
                with torch.cuda.stream(self.copy_stream):
                    # Intentionally blocking per chunk for strict comm-aware checkpoints.
                    dst.copy_(src, non_blocking=False)
                    chunk_done_event = torch.cuda.Event()
                    chunk_done_event.record(self.copy_stream)
                copy_dur_s = time.perf_counter() - copy_t0
            finally:
''',
     '''            try:
                copy_t0 = time.perf_counter()
                gate = self._stream_gate_for_copy()
                with torch.cuda.stream(self.copy_stream):
                    if gate is not None:
                        # GPU-gated asynchronous issue: the copy stream itself waits for
                        # the comm flag to be 0 before this chunk; the host never blocks.
                        self._gate_throttle_inflight()
                        gate.wait_zero(self.copy_stream)
                        chunk_start_event = torch.cuda.Event(enable_timing=True)
                        chunk_start_event.record(self.copy_stream)
                        dst.copy_(src, non_blocking=True)
                        chunk_done_event = torch.cuda.Event(enable_timing=True)
                        chunk_done_event.record(self.copy_stream)
                        self._gate_inflight.append(
                            (chunk_start_event, chunk_done_event, int(nbytes), bool(background))
                        )
                    else:
                        # Intentionally blocking per chunk for strict comm-aware checkpoints.
                        dst.copy_(src, non_blocking=False)
                        chunk_done_event = torch.cuda.Event()
                        chunk_done_event.record(self.copy_stream)
                copy_dur_s = time.perf_counter() - copy_t0
            finally:
'''),
    ('''            self._record_copy_stats(background=background, nbytes=nbytes, dur_s=copy_dur_s)
            if copy_dur_s > 0:
                inst_bw = nbytes / copy_dur_s
''',
     '''            if gate is not None:
                # Asynchronous issue: the host time above is the enqueue time; the copy
                # durations are read from the chunk events when they are reaped.
                self._gate_reap_inflight()
                copy_dur_s = 0.0
            else:
                self._record_copy_stats(background=background, nbytes=nbytes, dur_s=copy_dur_s)
            if copy_dur_s > 0:
                inst_bw = nbytes / copy_dur_s
'''),
    ('''    def quiesce_prefetch_for_comm(self, after_drain=None) -> None:
        if self.layerwise_offload_managers is None:
            if after_drain is not None:
                after_drain()
            return
        guards = []
''',
     '''    def quiesce_prefetch_for_comm(self, after_drain=None) -> None:
        if self.layerwise_offload_managers is None:
            if after_drain is not None:
                after_drain()
            return
        tracker = self.comm_activity_tracker
        if tracker is not None and getattr(tracker, "window_mode", None) == "gate":
            # GPU-gated prefetch: the copy stream pauses itself at the collective; no
            # host-side drain of the copy stream before the launch.
            if after_drain is not None:
                after_drain()
            return
        guards = []
'''),
])

apply(harness_path, "gpu|auto|gate)", [
    ("        launch|kernel|none|gpu|auto)\n", "        launch|kernel|none|gpu|auto|gate)\n"),
])
