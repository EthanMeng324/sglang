#!/usr/bin/env python3
"""Add comm window mode "gpu" (GPU-timed yield window) to CommunicationActivityTracker.

kernel mode marks the collective active when the host enqueues it, which is a queue
depth ahead of the GPU, so the prefetch worker is paused long before the GPU reaches
the collective. gpu mode records a CUDA event on the compute stream right before the
collective is launched and makes the window active only once that event has
completed on the GPU; the window ends when the NCCL Work object reports completion.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    "python/sglang/multimodal_gen/runtime/utils/comm_aware_prefetch.py"
)
src = path.read_text()
if "_gpu_region_loop" in src:
    print("already patched:", path)
    sys.exit(0)

edits = [
    (
        '''    if normalized in {"none", "off", "noyield"}:
        return "none"
    return "kernel"
''',
        '''    if normalized in {"none", "off", "noyield"}:
        return "none"
    if normalized in {"gpu", "device", "gpu_timed"}:
        return "gpu"
    return "kernel"
''',
    ),
    (
        '''class _KernelCommRegion:
    def __init__(self, tracker: "CommunicationActivityTracker", tag: str):
        self._tracker = tracker
        self._tag = tag
        self._end_event = torch.cuda.Event()
        self._finalized = False
        self._active_started = False
        self._nccl_work: Any = None

    def record_start(self) -> None:
        if self._active_started:
            return
        self._tracker._block_launches_start()
        self._active_started = True
''',
        '''class _KernelCommRegion:
    def __init__(
        self,
        tracker: "CommunicationActivityTracker",
        tag: str,
        gpu_timed: bool = False,
    ):
        self._tracker = tracker
        self._tag = tag
        self._end_event = torch.cuda.Event()
        self._finalized = False
        self._active_started = False
        self._nccl_work: Any = None
        self._gpu_timed = gpu_timed
        self._start_event: torch.cuda.Event | None = None

    def record_start(self) -> None:
        if self._active_started:
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
''',
    ),
    (
        '''        self._finalized = True
        if not self._active_started:
            return
        if self._nccl_work is not None:
            self._tracker._enqueue_kernel_region(
''',
        '''        self._finalized = True
        if not self._active_started:
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
''',
    ),
    (
        '''            self._work_thread.start()

    def uses_kernel_window(self) -> bool:
        return self._window_mode == "kernel"

    def begin_kernel_region(self, tag: str | None = None) -> _KernelCommRegion | None:
        if not self.uses_kernel_window() or not torch.cuda.is_available():
            return None
        return _KernelCommRegion(self, tag or "comm")
''',
        '''            self._work_thread.start()
        self._gpu_q: queue.Queue | None = None
        self._gpu_thread: threading.Thread | None = None
        if self._window_mode == "gpu" and torch.cuda.is_available():
            self._gpu_q = queue.Queue()
            if self._event_stop_event is None:
                self._event_stop_event = threading.Event()
            self._gpu_thread = threading.Thread(
                target=self._gpu_region_loop,
                name="CommActivityTrackerGpuRegions",
                daemon=True,
            )
            self._gpu_thread.start()

    def uses_kernel_window(self) -> bool:
        # Both modes launch the collective with async_op=True and track its Work.
        return self._window_mode in ("kernel", "gpu")

    def begin_kernel_region(self, tag: str | None = None) -> _KernelCommRegion | None:
        if not self.uses_kernel_window() or not torch.cuda.is_available():
            return None
        return _KernelCommRegion(
            self, tag or "comm", gpu_timed=(self._window_mode == "gpu")
        )

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
        self._gpu_q.put([False, start_event, nccl_work, end_event, tag])

    def _gpu_region_loop(self) -> None:
        """Open each region when its start event completes on the GPU and close it
        when its NCCL Work (or end event) completes. Regions are polled together so
        back-to-back collectives keep the window continuous."""
        assert self._gpu_q is not None
        assert self._event_stop_event is not None
        pending: list[list[Any]] = []
        while True:
            while True:
                try:
                    item = self._gpu_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    for it in pending:
                        if it[0]:
                            self._block_launches_end()
                    pending.clear()
                    return
                pending.append(item)

            progressed = False
            still: list[list[Any]] = []
            for item in pending:
                started, start_event, work, end_event, _tag = item
                if not started:
                    try:
                        reached = start_event.query()
                    except Exception:
                        reached = True
                    if not reached:
                        still.append(item)
                        continue
                    self._block_launches_start()
                    item[0] = True
                    progressed = True
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
                    self._block_launches_end()
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
''',
    ),
    (
        '''    def close(self) -> None:
        if self._event_stop_event is not None:
            self._event_stop_event.set()
        if self._work_q is not None:
            self._work_q.put(None)
''',
        '''    def close(self) -> None:
        if self._event_stop_event is not None:
            self._event_stop_event.set()
        if self._gpu_q is not None:
            self._gpu_q.put(None)
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=2.0)
            self._gpu_thread = None
        if self._work_q is not None:
            self._work_q.put(None)
''',
    ),
]
for old, new in edits:
    assert src.count(old) == 1, f"pattern not found exactly once:\n{old}"
    src = src.replace(old, new)
path.write_text(src)
print("patched:", path)
