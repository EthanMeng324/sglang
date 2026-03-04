import contextlib
import threading
import time
from typing import Any, Callable, Dict


class CommunicationActivityTracker:
    """Track whether communication is currently active.

    The tracker is framework-agnostic and only exposes a boolean activity signal.
    You can drive it manually with `comm_region()`, or patch torch.distributed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._active = 0
        self._patch_state: Dict[str, Any] | None = None

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
                self.mark_start(name)
                try:
                    result = fn(*args, **kwargs)
                    async_op = bool(kwargs.get("async_op", False))
                    if async_op and hasattr(result, "wait"):
                        return _WorkWrapper(result, self)
                    self.mark_end(name)
                    return result
                except Exception:
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
