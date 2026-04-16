import contextlib
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
    return "kernel"


class _KernelCommRegion:
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

    def uses_kernel_window(self) -> bool:
        return self._window_mode == "kernel"

    def begin_kernel_region(self, tag: str | None = None) -> _KernelCommRegion | None:
        if not self.uses_kernel_window() or not torch.cuda.is_available():
            return None
        return _KernelCommRegion(self, tag or "comm")

    def _block_launches_start(self) -> None:
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
        with self._cv:
            self._active += 1
            self._cv.notify_all()

    def mark_end(self, _tag: str | None = None) -> None:
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
        if self._work_q is not None:
            self._work_q.put(None)
        if self._work_thread is not None:
            self._work_thread.join(timeout=2.0)
            self._work_thread = None
        if self._event_thread is not None:
            self._event_thread.join(timeout=1.0)
            self._event_thread = None
