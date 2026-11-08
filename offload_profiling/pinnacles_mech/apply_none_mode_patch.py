#!/usr/bin/env python3
"""Add comm window mode "none" to CommunicationActivityTracker.

SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE=none keeps chunked prefetch but never marks a
collective as active, so the prefetch worker never yields ("chunked, no yield").
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    "python/sglang/multimodal_gen/runtime/utils/comm_aware_prefetch.py"
)
src = path.read_text()
if '"none"' in src and "noyield" in src:
    print("already patched:", path)
    sys.exit(0)

edits = [
    (
        '''    if normalized in {"launch", "host", "wrapper"}:
        return "launch"
    return "kernel"
''',
        '''    if normalized in {"launch", "host", "wrapper"}:
        return "launch"
    if normalized in {"none", "off", "noyield"}:
        return "none"
    return "kernel"
''',
    ),
    (
        '''    def _block_launches_start(self) -> None:
        with self._cv:
''',
        '''    def _block_launches_start(self) -> None:
        if self._window_mode == "none":
            return
        with self._cv:
''',
    ),
    (
        '''    def mark_start(self, _tag: str | None = None) -> None:
        with self._cv:
''',
        '''    def mark_start(self, _tag: str | None = None) -> None:
        if self._window_mode == "none":
            return
        with self._cv:
''',
    ),
    (
        '''    def mark_end(self, _tag: str | None = None) -> None:
        with self._cv:
''',
        '''    def mark_end(self, _tag: str | None = None) -> None:
        if self._window_mode == "none":
            return
        with self._cv:
''',
    ),
]
for old, new in edits:
    assert src.count(old) == 1, f"pattern not found exactly once:\n{old}"
    src = src.replace(old, new)
path.write_text(src)
print("patched:", path)
