#!/usr/bin/env python3
"""Rebuttal baseline (reviewer PShp W1, alternative 2): "pausing before coarse-grained communication".
Reuses the GPU-side gate but keeps the flag set for a whole communication phase: it is written to 1 before the
first (input) collective of an attention and only cleared after the attention's output collective completes,
so the prefetch also stays paused during the attention kernels in between. SGLANG_DIT_COMM_GATE_COARSE=1
with window mode "gate". A safety valve clears the flag after 8 unclosed input collectives.
usage: apply_coarse_patch.py [repo_root]"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
path = root / "python/sglang/multimodal_gen/runtime/utils/comm_aware_prefetch.py"
src = path.read_text()
if "_coarse_phase_done" in src:
    print("already patched:", path)
    sys.exit(0)
edits = [
    ('''        if self._gate is not None:
            # The current stream already waits on the collective's Work, so this
            # write lands right after the collective completes on the GPU.
            self._gate.write(torch.cuda.current_stream(), 0)
            return
''',
     '''        if self._gate is not None:
            if self._tracker._gate_coarse and not self._tracker._coarse_phase_done(self._tag):
                # coarse-grained pause (rebuttal baseline): the flag stays set from the
                # first collective of an attention phase until its output collective
                return
            # The current stream already waits on the collective's Work, so this
            # write lands right after the collective completes on the GPU.
            self._gate.write(torch.cuda.current_stream(), 0)
            return
'''),
    ('''        self._gate: _StreamGate | None = None
        self._gate_failed = False
''',
     '''        self._gate: _StreamGate | None = None
        self._gate_failed = False
        # coarse-grained pause: keep the flag set across a whole attention phase
        self._gate_coarse = os.getenv("SGLANG_DIT_COMM_GATE_COARSE", "0").strip().lower() in {"1", "true", "yes"}
        self._coarse_open = 0
'''),
    ('''    def gate_stats(self) -> Dict[str, int]:
''',
     '''    def _coarse_phase_done(self, tag: str) -> bool:
        """True when this collective closes a coarse communication phase (the output
        collective of an attention), or as a safety valve after 8 unclosed inputs."""
        t = (tag or "").lower()
        if "output" in t or "all_gather" in t:
            self._coarse_open = 0
            return True
        self._coarse_open += 1
        if self._coarse_open > 8:
            logger.warning("[comm-gate] coarse phase not closed after 8 collectives; clearing the flag")
            self._coarse_open = 0
            return True
        return False

    def gate_stats(self) -> Dict[str, int]:
'''),
]
for old, new in edits:
    assert src.count(old) == 1, f"pattern not found exactly once:\n{old}"
    src = src.replace(old, new)
path.write_text(src)
print("patched:", path)
