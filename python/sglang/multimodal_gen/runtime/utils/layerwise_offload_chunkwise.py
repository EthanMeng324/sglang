import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import torch

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.comm_aware_prefetch import (
    CommunicationActivityTracker,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return v


def _parse_phase_name_csv(raw: str | None) -> Set[str]:
    if raw is None:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    prefixes: Tuple[str, ...]


class LayerwiseOffloadManager:
    """Simplified chunk-wise, comm-aware layerwise offload manager.

    This implementation intentionally starts from the original offload logic and
    keeps only the minimal extra machinery required by chunk-wise prefetch:
    - A background prefetch thread (asynchronous to the compute thread).
    - Chunk-by-chunk H2D copy on a dedicated copy stream.
    - Communication-aware pause/resume before each chunk launch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layers_attr_str: str,
        num_layers: int,
        enabled: bool,
        pin_cpu_memory: bool = True,
        prefetch_size: int = 1,
        comm_aware: bool = False,
        comm_patch_torch_distributed: bool = False,
        prefetch_chunk_size_mb: int = 32,
        phase_specs: Sequence[PhaseSpec] | None = None,
        resident_phase_names: Set[str] | None = None,
        submodule_granularity: bool = True,
    ) -> None:
        self.model = model
        self.layers_attr_str = layers_attr_str
        self.num_layers = num_layers
        self.pin_cpu_memory = pin_cpu_memory
        self.prefetch_size = min(max(1, prefetch_size), self.num_layers)
        self.comm_aware = comm_aware
        self.comm_patch_torch_distributed = comm_patch_torch_distributed
        self.prefetch_chunk_size_bytes = max(1, prefetch_chunk_size_mb) * 1024 * 1024
        self.submodule_granularity = submodule_granularity
        self.phase_specs: Tuple[PhaseSpec, ...] = tuple(
            phase_specs or (PhaseSpec(name="layer", prefixes=("",)),)
        )
        self.phase_name_to_idx = {
            spec.name: phase_idx for phase_idx, spec in enumerate(self.phase_specs)
        }
        self.resident_phase_names = {
            name
            for name in (resident_phase_names or set())
            if name in self.phase_name_to_idx
        }
        self._resident_phase_ids = {
            self.phase_name_to_idx[name] for name in self.resident_phase_names
        }
        self.enabled = bool(enabled and torch.cuda.is_available())

        self.comm_tracker: CommunicationActivityTracker | None = None
        if not self.enabled:
            return

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.copy_stream = torch.cuda.Stream()
        self.comm_tracker = CommunicationActivityTracker() if self.comm_aware else None

        self._layer_name_re = re.compile(
            rf"(^|\.){re.escape(layers_attr_str)}\.(\d+)(\.|$)"
        )
        self._unmatched_phase_suffixes: Set[str] = set()

        # layer_idx -> {phase_idx: {dtype: consolidated pinned CPU buffer}}
        self._consolidated_cpu_weights: Dict[
            int, Dict[int, Dict[torch.dtype, torch.Tensor]]
        ] = {}
        # layer_idx -> total bytes for profiling
        self._layer_total_bytes: Dict[int, int] = {}
        # layer_idx -> {phase_idx: phase_prefetch_bytes}
        self._phase_total_bytes: Dict[int, Dict[int, int]] = {}
        # layer_idx -> {name: {dtype, phase_id, offset, numel, shape, resident}}
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}

        # layer_idx -> {phase_idx: {dtype: phase-sized GPU buffer}}
        self._prefetch_gpu_buffers: Dict[
            int, Dict[int, Dict[torch.dtype, torch.Tensor]]
        ] = {}
        # layer_idx -> {phase_idx: {dtype: copied_numel}}
        self._prefetch_offsets: Dict[int, Dict[int, Dict[torch.dtype, int]]] = {}

        # layer_idx -> event recorded on copy stream when prefetch completes
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        # layer_idx -> {phase_idx: event recorded on copy stream when phase completes}
        self._phase_events: Dict[int, Dict[int, torch.cuda.Event]] = {}
        # layer_idx -> {phase_idx: [(chunk_done_event, chunk_nbytes)]}
        self._prefetch_chunk_events: Dict[
            int, Dict[int, List[Tuple[torch.cuda.Event, int]]]
        ] = {}
        # GPU resident layers
        self._gpu_layers: Set[int] = set()

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}
        self._forward_hooks: List[Any] = []

        # Background prefetch scheduling state (intentionally simple).
        self._state_lock = threading.RLock()
        self._copy_lock = threading.Lock()
        self._scheduled_layers: Set[int] = set()
        self._anchor_layer = 0
        self._urgent_layer: int | None = None

        self._worker_thread: threading.Thread | None = None
        self._worker_stop_event = threading.Event()
        self._worker_wakeup_event = threading.Event()
        self._logical_step_idx: int | None = None

        # Per-step lightweight diagnostics kept from chunkwise profiling.
        self._step_seq = 0
        self._step_active = False
        self._step_wall_start = 0.0
        self._step_bg_chunks = 0
        self._step_bg_bytes = 0
        self._step_bg_copy_s = 0.0
        self._step_sync_chunks = 0
        self._step_sync_bytes = 0
        self._step_sync_copy_s = 0.0
        self._step_wait_comm_bg_s = 0.0
        self._step_wait_comm_sync_s = 0.0
        self._step_ensure_ready_s = 0.0
        self._step_unique_loaded_bytes = 0
        self._step_unique_loaded_layers: Set[int] = set()
        self._prefetch_cp_wait_pairs: List[
            Tuple[int, int, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._prefetch_wait_byte_records: List[Tuple[int, int, int]] = []

        self._initialize()

    def _match_layer_idx(self, name: str) -> int | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    def _relative_name_after_layer(self, name: str) -> str | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        return name[m.end() :]

    def _match_phase_idx(self, name: str) -> int:
        relative_name = self._relative_name_after_layer(name) or ""
        for phase_idx, spec in enumerate(self.phase_specs):
            for prefix in spec.prefixes:
                if not prefix:
                    return phase_idx
                if relative_name == prefix or relative_name.startswith(prefix + "."):
                    return phase_idx

        fallback_idx = len(self.phase_specs) - 1
        if relative_name not in self._unmatched_phase_suffixes:
            self._unmatched_phase_suffixes.add(relative_name)
            logger.warning(
                "Offload phase matcher for %s fell back to phase %s on tensor %s.",
                self.layers_attr_str,
                self.phase_specs[fallback_idx].name,
                relative_name,
            )
        return fallback_idx

    def _phase_prefetch_bytes(self, layer_idx: int, phase_idx: int) -> int:
        return int(self._phase_total_bytes.get(layer_idx, {}).get(phase_idx, 0))

    def _phase_is_resident(self, phase_idx: int) -> bool:
        return phase_idx in self._resident_phase_ids

    def _phase_ready_locked(self, layer_idx: int, phase_idx: int) -> bool:
        if self._phase_is_resident(phase_idx):
            return True
        if self._phase_prefetch_bytes(layer_idx, phase_idx) <= 0:
            return True
        return phase_idx in self._phase_events.get(layer_idx, {})

    def _layer_complete_locked(self, layer_idx: int) -> bool:
        for phase_idx in range(len(self.phase_specs)):
            if not self._phase_ready_locked(layer_idx, phase_idx):
                return False
        return True

    def _first_incomplete_phase_locked(self, layer_idx: int) -> int | None:
        for phase_idx in range(len(self.phase_specs)):
            if not self._phase_ready_locked(layer_idx, phase_idx):
                return phase_idx
        return None

    def _target_phase_event_locked(
        self, layer_idx: int, phase_idx: int
    ) -> torch.cuda.Event | None:
        if self._phase_is_resident(phase_idx):
            return None
        if self._phase_prefetch_bytes(layer_idx, phase_idx) <= 0:
            return None
        return self._phase_events.get(layer_idx, {}).get(phase_idx)

    def _mark_layer_complete_locked(self, layer_idx: int) -> None:
        if layer_idx in self._prefetch_events:
            return
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        self._prefetch_events[layer_idx] = event
        self._gpu_layers.add(layer_idx)
        self._scheduled_layers.discard(layer_idx)
        if self._urgent_layer == layer_idx:
            self._urgent_layer = None
        self._record_unique_layer_loaded(layer_idx)

    def _begin_step_stats(self) -> None:
        with self._state_lock:
            if self._step_active:
                return
            self._step_seq += 1
            self._step_active = True
            self._step_wall_start = time.perf_counter()
            self._step_bg_chunks = 0
            self._step_bg_bytes = 0
            self._step_bg_copy_s = 0.0
            self._step_sync_chunks = 0
            self._step_sync_bytes = 0
            self._step_sync_copy_s = 0.0
            self._step_wait_comm_bg_s = 0.0
            self._step_wait_comm_sync_s = 0.0
            self._step_ensure_ready_s = 0.0
            self._step_unique_loaded_bytes = 0
            self._step_unique_loaded_layers = set()

    def _record_copy_stats(self, *, background: bool, nbytes: int, dur_s: float) -> None:
        with self._state_lock:
            if not self._step_active:
                return
            if background:
                self._step_bg_chunks += 1
                self._step_bg_bytes += int(nbytes)
                self._step_bg_copy_s += float(dur_s)
            else:
                self._step_sync_chunks += 1
                self._step_sync_bytes += int(nbytes)
                self._step_sync_copy_s += float(dur_s)

    def _record_wait_comm_stats(self, *, background: bool, dur_s: float) -> None:
        with self._state_lock:
            if not self._step_active:
                return
            if background:
                self._step_wait_comm_bg_s += float(dur_s)
            else:
                self._step_wait_comm_sync_s += float(dur_s)

    def _record_unique_layer_loaded(self, layer_idx: int) -> None:
        with self._state_lock:
            if not self._step_active:
                return
            if layer_idx in self._step_unique_loaded_layers:
                return
            self._step_unique_loaded_layers.add(layer_idx)
            self._step_unique_loaded_bytes += int(self._layer_total_bytes.get(layer_idx, 0))

    def _record_ensure_ready_stats(self, dur_s: float) -> None:
        with self._state_lock:
            if not self._step_active:
                return
            self._step_ensure_ready_s += float(dur_s)

    @torch.compiler.disable
    def _current_profile_step_idx(self) -> int:
        if self._logical_step_idx is not None:
            return self._logical_step_idx
        return max(0, self._step_seq - 1)

    @torch.compiler.disable
    def set_logical_step_idx(self, step_idx: int | None) -> None:
        self._logical_step_idx = step_idx

    @torch.compiler.disable
    def _record_prefetch_cp_wait_start(self) -> torch.cuda.Event | None:
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream())
        return event

    @torch.compiler.disable
    def _record_prefetch_cp_wait_end(
        self, step_idx: int, layer_idx: int, start_event: torch.cuda.Event | None
    ) -> None:
        if not self.enabled or start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(torch.cuda.current_stream())
        self._prefetch_cp_wait_pairs.append(
            (step_idx, layer_idx, start_event, end_event)
        )

    @torch.compiler.disable
    def _estimate_waited_prefetch_bytes(
        self, layer_idx: int, phase_idx: int | None = None
    ) -> Tuple[int, int]:
        total_bytes = (
            self._phase_prefetch_bytes(layer_idx, phase_idx)
            if phase_idx is not None
            else int(self._layer_total_bytes.get(layer_idx, 0))
        )
        if total_bytes <= 0:
            return 0, 0

        with self._state_lock:
            layer_known = layer_idx in self._consolidated_cpu_weights
            if phase_idx is None:
                if layer_idx in self._gpu_layers:
                    ready_event = self._prefetch_events.get(layer_idx)
                    if ready_event is None or ready_event.query():
                        return 0, total_bytes
                phase_event_dict = self._phase_events.get(layer_idx, {})
                chunk_events_by_phase = dict(
                    self._prefetch_chunk_events.get(layer_idx, {})
                )
            else:
                if self._phase_is_resident(phase_idx):
                    return 0, total_bytes
                phase_event = self._phase_events.get(layer_idx, {}).get(phase_idx)
                if phase_event is not None and phase_event.query():
                    return 0, total_bytes
                chunk_events_by_phase = {
                    phase_idx: list(
                        self._prefetch_chunk_events.get(layer_idx, {}).get(
                            phase_idx, []
                        )
                    )
                }

        if not layer_known:
            return 0, 0
        if not chunk_events_by_phase:
            return total_bytes, total_bytes

        completed_bytes = 0
        for current_phase_idx, chunk_events in chunk_events_by_phase.items():
            if phase_idx is None:
                phase_event = phase_event_dict.get(current_phase_idx)
                if phase_event is not None and phase_event.query():
                    completed_bytes += self._phase_prefetch_bytes(
                        layer_idx, current_phase_idx
                    )
                    continue
            for event, nbytes in chunk_events:
                try:
                    if event.query():
                        completed_bytes += int(nbytes)
                except Exception:
                    continue

        waited_bytes = max(0, total_bytes - completed_bytes)
        return waited_bytes, total_bytes

    @torch.compiler.disable
    def _record_waited_prefetch_bytes(
        self, step_idx: int, waited_bytes: int, total_bytes: int
    ) -> None:
        if total_bytes <= 0:
            return
        self._prefetch_wait_byte_records.append(
            (step_idx, int(max(0, waited_bytes)), int(max(0, total_bytes)))
        )

    @torch.compiler.disable
    def collect_profile_metrics(self) -> Dict[str, List[float]]:
        if not self.enabled or self.device is None:
            return {}
        if self._prefetch_cp_wait_pairs:
            torch.cuda.synchronize(self.device)

        per_step_ms: Dict[int, float] = {}
        per_step_layers: Dict[int, float] = {}
        per_step_waited_bytes: Dict[int, float] = {}
        per_step_total_bytes: Dict[int, float] = {}
        event_errors = 0
        for step_idx, _layer_idx, start_event, end_event in self._prefetch_cp_wait_pairs:
            try:
                wait_ms = max(0.0, float(start_event.elapsed_time(end_event)))
            except Exception:
                event_errors += 1
                continue
            per_step_ms[step_idx] = per_step_ms.get(step_idx, 0.0) + wait_ms
            if wait_ms > 0.01:
                per_step_layers[step_idx] = per_step_layers.get(step_idx, 0.0) + 1.0
        for step_idx, waited_bytes, total_bytes in self._prefetch_wait_byte_records:
            per_step_waited_bytes[step_idx] = (
                per_step_waited_bytes.get(step_idx, 0.0) + float(waited_bytes)
            )
            per_step_total_bytes[step_idx] = (
                per_step_total_bytes.get(step_idx, 0.0) + float(total_bytes)
            )

        max_step = max(
            [-1]
            + list(per_step_ms.keys())
            + list(per_step_layers.keys())
            + list(per_step_waited_bytes.keys())
            + list(per_step_total_bytes.keys())
        )
        if max_step < 0:
            return {}

        wait_ms_list = [0.0] * (max_step + 1)
        wait_layers_list = [0.0] * (max_step + 1)
        waited_bytes_list = [0.0] * (max_step + 1)
        total_bytes_list = [0.0] * (max_step + 1)
        for step_idx, value in per_step_ms.items():
            wait_ms_list[step_idx] = value
        for step_idx, value in per_step_layers.items():
            wait_layers_list[step_idx] = value
        for step_idx, value in per_step_waited_bytes.items():
            waited_bytes_list[step_idx] = value
        for step_idx, value in per_step_total_bytes.items():
            total_bytes_list[step_idx] = value

        if self._prefetch_wait_byte_records and not self._prefetch_cp_wait_pairs:
            logger.warning(
                "Chunkwise offload profiling captured waited-byte records but no CUDA "
                "event pairs; exporting zero-filled prefetch_critical_path_wait_ms."
            )
        elif event_errors > 0:
            logger.warning(
                "Chunkwise offload profiling dropped %d invalid CUDA event pairs.",
                event_errors,
            )

        return {
            "prefetch_critical_path_wait_ms": wait_ms_list,
            "prefetch_critical_path_wait_layers": wait_layers_list,
            "prefetch_critical_path_waited_bytes": waited_bytes_list,
            "prefetch_critical_path_total_bytes": total_bytes_list,
        }

    def _end_step_stats(self) -> None:
        with self._state_lock:
            if not self._step_active:
                return
            step = self._step_seq
            wall_s = max(0.0, time.perf_counter() - self._step_wall_start)
            bg_chunks = self._step_bg_chunks
            bg_mb = self._step_bg_bytes / (1024.0 * 1024.0)
            bg_copy_s = self._step_bg_copy_s
            sync_chunks = self._step_sync_chunks
            sync_mb = self._step_sync_bytes / (1024.0 * 1024.0)
            sync_copy_s = self._step_sync_copy_s
            wait_bg_s = self._step_wait_comm_bg_s
            wait_sync_s = self._step_wait_comm_sync_s
            ensure_s = self._step_ensure_ready_s
            unique_mb = self._step_unique_loaded_bytes / (1024.0 * 1024.0)
            unique_layers = len(self._step_unique_loaded_layers)
            self._step_active = False

        total_mb = bg_mb + sync_mb
        sync_ratio = 0.0 if total_mb <= 0 else (100.0 * sync_mb / total_mb)
        dup_ratio = 0.0 if unique_mb <= 0 else (total_mb / unique_mb)
        logger.info(
            "[ChunkwiseStep] "
            f"step={step} wall_s={wall_s:.3f} "
            f"bg_chunks={bg_chunks} bg_mb={bg_mb:.2f} bg_copy_s={bg_copy_s:.3f} "
            f"sync_chunks={sync_chunks} sync_mb={sync_mb:.2f} sync_copy_s={sync_copy_s:.3f} "
            f"sync_ratio_mb_pct={sync_ratio:.2f} "
            f"wait_comm_bg_s={wait_bg_s:.3f} wait_comm_sync_s={wait_sync_s:.3f} "
            f"ensure_ready_s={ensure_s:.3f} "
            f"unique_loaded_layers={unique_layers} unique_loaded_mb={unique_mb:.2f} "
            f"copy_dup_ratio={dup_ratio:.3f}"
        )

    def _schedule_layer_locked(self, layer_idx: int) -> bool:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return False
        if layer_idx not in self._consolidated_cpu_weights:
            return False
        if self._layer_complete_locked(layer_idx):
            return False
        if layer_idx in self._scheduled_layers:
            return False
        self._scheduled_layers.add(layer_idx)
        return True

    def _pick_next_phase_locked(self) -> Tuple[int, int] | None:
        stale = [
            layer_idx
            for layer_idx in self._scheduled_layers
            if (
                layer_idx not in self._consolidated_cpu_weights
                or self._layer_complete_locked(layer_idx)
            )
        ]
        for layer_idx in stale:
            self._scheduled_layers.discard(layer_idx)

        if self._urgent_layer is not None and self._urgent_layer in self._scheduled_layers:
            phase_idx = self._first_incomplete_phase_locked(self._urgent_layer)
            if phase_idx is not None:
                return self._urgent_layer, phase_idx

        if not self._scheduled_layers:
            return None

        anchor = self._anchor_layer
        layer_idx = min(
            self._scheduled_layers,
            key=lambda layer_idx: (layer_idx - anchor) % self.num_layers,
        )
        phase_idx = self._first_incomplete_phase_locked(layer_idx)
        if phase_idx is None:
            self._scheduled_layers.discard(layer_idx)
            return None
        return layer_idx, phase_idx

    def _schedule_prefetch_window(self, anchor: int) -> None:
        queued = False
        with self._state_lock:
            self._anchor_layer = anchor
            for j in range(anchor + self.prefetch_size, anchor + 2 * self.prefetch_size):
                layer_idx = j % self.num_layers
                queued = self._schedule_layer_locked(layer_idx) or queued
        if queued:
            self._worker_wakeup_event.set()

    def _wait_for_comm_inactive(
        self,
        *,
        background: bool,
        stop_event: threading.Event | None = None,
        timeout: float = 0.001,
    ) -> bool:
        if self.comm_tracker is None or not self.comm_tracker.is_active():
            return True

        marker = "SGL_PREFETCH_WAIT_COMM_BG" if background else "SGL_PREFETCH_WAIT_COMM_SYNC"
        push_nvtx = bool(torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"))
        if push_nvtx:
            torch.cuda.nvtx.range_push(marker)
        wait_t0 = time.perf_counter()
        try:
            return self.comm_tracker.wait_until_inactive(
                stop_event=stop_event,
                timeout=timeout,
            )
        finally:
            self._record_wait_comm_stats(
                background=background,
                dur_s=time.perf_counter() - wait_t0,
            )
            if push_nvtx:
                torch.cuda.nvtx.range_pop()

    def _start_prefetch_worker(self) -> None:
        if not self.enabled:
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._worker_stop_event.clear()
        self._worker_wakeup_event.set()
        self._worker_thread = threading.Thread(
            target=self._prefetch_worker_loop,
            name=f"LayerwisePrefetchWorker-{self.layers_attr_str}",
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_prefetch_worker(self) -> None:
        if self._worker_thread is None:
            return

        self._worker_stop_event.set()
        self._worker_wakeup_event.set()
        self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    def _prefetch_worker_loop(self) -> None:
        if self.device is not None:
            torch.cuda.set_device(self.device)

        while not self._worker_stop_event.is_set():
            with self._state_lock:
                next_phase = self._pick_next_phase_locked()

            if next_phase is None:
                self._worker_wakeup_event.wait(timeout=0.01)
                self._worker_wakeup_event.clear()
                continue
            layer_idx, phase_idx = next_phase

            if not self._wait_for_comm_inactive(
                background=True,
                stop_event=self._worker_stop_event,
                timeout=None,
            ):
                continue

            copied = self._copy_one_chunk(layer_idx, phase_idx, background=True)
            if not copied:
                time.sleep(0.0005)

    def _bind_phase_weights_locked(self, layer_idx: int, phase_idx: int) -> None:
        gpu_buffers = self._prefetch_gpu_buffers[layer_idx][phase_idx]
        for name, meta in self._weight_metadata[layer_idx].items():
            if meta["phase_id"] != phase_idx or meta["resident"]:
                continue
            dtype = meta["dtype"]
            gpu_buffer = gpu_buffers[dtype]
            target = self.get_target_with_name(name)
            target.data = gpu_buffer[
                meta["offset"] : meta["offset"] + meta["numel"]
            ].view(meta["shape"])

    @torch.compiler.disable
    def quiesce_copy_stream_for_comm(self) -> None:
        """Ensure no in-flight/offloaded H2D remains before communication starts."""
        if not self.enabled or self.copy_stream is None:
            return
        with self._copy_lock:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    @torch.compiler.disable
    def _copy_one_chunk(
        self, layer_idx: int, phase_idx: int, *, background: bool
    ) -> bool:
        if not self.enabled or self.device is None or self.copy_stream is None:
            return False

        with self._copy_lock:
            with self._state_lock:
                if self._phase_ready_locked(layer_idx, phase_idx):
                    if self._layer_complete_locked(layer_idx):
                        self._scheduled_layers.discard(layer_idx)
                        if self._urgent_layer == layer_idx:
                            self._urgent_layer = None
                    return False

                if layer_idx not in self._consolidated_cpu_weights:
                    self._scheduled_layers.discard(layer_idx)
                    if self._urgent_layer == layer_idx:
                        self._urgent_layer = None
                    return False

                if self._phase_prefetch_bytes(layer_idx, phase_idx) <= 0:
                    if layer_idx not in self._phase_events:
                        self._phase_events[layer_idx] = {}
                    if phase_idx not in self._phase_events[layer_idx]:
                        event = torch.cuda.Event()
                        event.record(self.copy_stream)
                        self._phase_events[layer_idx][phase_idx] = event
                    if self._layer_complete_locked(layer_idx):
                        self._mark_layer_complete_locked(layer_idx)
                    return False

                self._prefetch_gpu_buffers.setdefault(layer_idx, {})
                self._prefetch_offsets.setdefault(layer_idx, {})
                self._phase_events.setdefault(layer_idx, {})
                self._prefetch_chunk_events.setdefault(layer_idx, {})

                if phase_idx not in self._prefetch_gpu_buffers[layer_idx]:
                    cpu_phase_buffers = self._consolidated_cpu_weights[layer_idx].get(
                        phase_idx, {}
                    )
                    self._prefetch_gpu_buffers[layer_idx][phase_idx] = {
                        dtype: torch.empty(
                            cpu_buffer.shape,
                            dtype=dtype,
                            device=self.device,
                        )
                        for dtype, cpu_buffer in cpu_phase_buffers.items()
                    }
                    self._prefetch_offsets[layer_idx][phase_idx] = {
                        dtype: 0 for dtype in cpu_phase_buffers
                    }
                    self._bind_phase_weights_locked(layer_idx, phase_idx)

                offsets = self._prefetch_offsets[layer_idx][phase_idx]
                cpu_by_dtype = self._consolidated_cpu_weights[layer_idx].get(
                    phase_idx, {}
                )
                gpu_by_dtype = self._prefetch_gpu_buffers[layer_idx][phase_idx]

                chosen_dtype = None
                chosen_start = 0
                chosen_end = 0
                for dtype, cpu_buffer in cpu_by_dtype.items():
                    start = offsets[dtype]
                    if start >= cpu_buffer.numel():
                        continue
                    chunk_numel = max(
                        1,
                        self.prefetch_chunk_size_bytes // cpu_buffer.element_size(),
                    )
                    end = min(cpu_buffer.numel(), start + chunk_numel)
                    chosen_dtype = dtype
                    chosen_start = start
                    chosen_end = end
                    break

                if chosen_dtype is None:
                    if phase_idx not in self._phase_events[layer_idx]:
                        event = torch.cuda.Event()
                        event.record(self.copy_stream)
                        self._phase_events[layer_idx][phase_idx] = event
                    if self._layer_complete_locked(layer_idx):
                        self._mark_layer_complete_locked(layer_idx)
                    return False

                src = cpu_by_dtype[chosen_dtype][chosen_start:chosen_end]
                dst = gpu_by_dtype[chosen_dtype][chosen_start:chosen_end]
                nbytes = (chosen_end - chosen_start) * src.element_size()

            push_nvtx = bool(torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"))
            if push_nvtx:
                torch.cuda.nvtx.range_push(
                    "SGL_PREFETCH_H2D_BG" if background else "SGL_PREFETCH_H2D_SYNC"
                )
                torch.cuda.nvtx.range_push("SGL_PREFETCH_H2D")
            try:
                copy_t0 = time.perf_counter()
                with torch.cuda.stream(self.copy_stream):
                    # Intentionally blocking per chunk for strict comm-aware checkpoints.
                    dst.copy_(src, non_blocking=False)
                    chunk_done_event = torch.cuda.Event()
                    chunk_done_event.record(self.copy_stream)
                copy_dur_s = time.perf_counter() - copy_t0
            finally:
                if push_nvtx:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()

            self._record_copy_stats(background=background, nbytes=nbytes, dur_s=copy_dur_s)

            with self._state_lock:
                self._prefetch_chunk_events.setdefault(layer_idx, {}).setdefault(
                    phase_idx, []
                ).append(
                    (chunk_done_event, int(nbytes))
                )
                if (
                    layer_idx not in self._prefetch_offsets
                    or phase_idx not in self._prefetch_offsets[layer_idx]
                ):
                    return True

                self._prefetch_offsets[layer_idx][phase_idx][chosen_dtype] = chosen_end

                complete = all(
                    self._prefetch_offsets[layer_idx][phase_idx][dtype]
                    >= self._consolidated_cpu_weights[layer_idx][phase_idx][
                        dtype
                    ].numel()
                    for dtype in self._consolidated_cpu_weights[layer_idx].get(
                        phase_idx, {}
                    )
                )
                if complete and phase_idx not in self._phase_events[layer_idx]:
                    event = torch.cuda.Event()
                    event.record(self.copy_stream)
                    self._phase_events[layer_idx][phase_idx] = event
                if self._layer_complete_locked(layer_idx):
                    self._mark_layer_complete_locked(layer_idx)

            return True

    @torch.compiler.disable
    def _prefetch_layer_blocking(
        self, layer_idx: int, target_phase_idx: int | None = None
    ) -> None:
        if not self.enabled or self.device is None:
            return

        with self._state_lock:
            self._schedule_layer_locked(layer_idx)
            self._urgent_layer = layer_idx
        self._worker_wakeup_event.set()

        while True:
            with self._state_lock:
                has_data = layer_idx in self._consolidated_cpu_weights
                if target_phase_idx is None:
                    done = self._layer_complete_locked(layer_idx)
                    event = self._prefetch_events.get(layer_idx)
                else:
                    done = self._phase_ready_locked(layer_idx, target_phase_idx)
                    event = self._target_phase_event_locked(layer_idx, target_phase_idx)
            if done:
                if event is not None:
                    torch.cuda.current_stream().wait_event(event)
                return
            if not has_data:
                return

            if not self._wait_for_comm_inactive(background=False, timeout=None):
                continue

            with self._state_lock:
                frontier_phase_idx = self._first_incomplete_phase_locked(layer_idx)
            if frontier_phase_idx is None:
                time.sleep(0.0005)
                continue

            copied = self._copy_one_chunk(
                layer_idx, frontier_phase_idx, background=False
            )
            if copied:
                continue

            time.sleep(0.0005)

    @torch.compiler.disable
    def _ensure_phase_ready(
        self, layer_idx: int, target_phase_idx: int | None = None
    ) -> None:
        if not self.enabled or self.device is None:
            return

        ensure_t0 = time.perf_counter()
        push_nvtx = bool(torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"))
        if push_nvtx:
            torch.cuda.nvtx.range_push(
                "SGL_PREFETCH_ENSURE_PHASE_READY"
                if target_phase_idx is not None
                else "SGL_PREFETCH_ENSURE_READY"
            )
        try:
            with self._state_lock:
                self._schedule_layer_locked(layer_idx)
                self._urgent_layer = layer_idx
            self._worker_wakeup_event.set()

            while True:
                with self._state_lock:
                    if target_phase_idx is None:
                        ready = self._layer_complete_locked(layer_idx)
                        event = self._prefetch_events.get(layer_idx)
                    else:
                        ready = self._phase_ready_locked(layer_idx, target_phase_idx)
                        event = self._target_phase_event_locked(
                            layer_idx, target_phase_idx
                        )
                if ready:
                    break

                worker_alive = (
                    self._worker_thread is not None and self._worker_thread.is_alive()
                )
                if not worker_alive:
                    self._prefetch_layer_blocking(
                        layer_idx, target_phase_idx=target_phase_idx
                    )
                    with self._state_lock:
                        if target_phase_idx is None:
                            ready = self._layer_complete_locked(layer_idx)
                            event = self._prefetch_events.get(layer_idx)
                        else:
                            ready = self._phase_ready_locked(
                                layer_idx, target_phase_idx
                            )
                            event = self._target_phase_event_locked(
                                layer_idx, target_phase_idx
                            )
                    if ready:
                        break

                self._worker_wakeup_event.set()
                time.sleep(0.0005)

            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            with self._state_lock:
                if target_phase_idx is None and self._layer_complete_locked(layer_idx):
                    self._gpu_layers.add(layer_idx)
                if self._urgent_layer == layer_idx and (
                    target_phase_idx is None
                    or self._phase_ready_locked(layer_idx, target_phase_idx)
                ):
                    self._urgent_layer = None
        finally:
            if push_nvtx:
                torch.cuda.nvtx.range_pop()
            self._record_ensure_ready_stats(time.perf_counter() - ensure_t0)

    @torch.compiler.disable
    def _initialize(self) -> None:
        if not self.enabled:
            return

        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        # 1. collect and group tensors by layer / phase / dtype
        layer_groups: Dict[
            int, Dict[int, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]]
        ] = {}
        all_tensors = chain(self._named_parameters.items(), self._named_buffers.items())
        for name, tensor in all_tensors:
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            phase_idx = self._match_phase_idx(name)
            layer_groups.setdefault(layer_idx, {}).setdefault(phase_idx, {}).setdefault(
                tensor.dtype, []
            ).append((name, tensor))

        # 2. concat and offload (in pinned memory)
        for layer_idx, phase_to_dtype_params in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}
            self._layer_total_bytes[layer_idx] = 0
            self._phase_total_bytes[layer_idx] = {
                phase_idx: 0 for phase_idx in range(len(self.phase_specs))
            }

            for phase_idx, dtype_to_params in phase_to_dtype_params.items():
                self._consolidated_cpu_weights[layer_idx][phase_idx] = {}
                resident_phase = self._phase_is_resident(phase_idx)

                for dtype, weights in dtype_to_params.items():
                    total_numel = sum(t.numel() for _, t in weights)
                    phase_bytes = int(total_numel * weights[0][1].element_size())
                    if not resident_phase:
                        self._layer_total_bytes[layer_idx] += phase_bytes
                        self._phase_total_bytes[layer_idx][phase_idx] += phase_bytes

                    cpu_buffer = torch.empty(
                        total_numel,
                        dtype=dtype,
                        pin_memory=self.pin_cpu_memory,
                    )

                    current_offset = 0
                    for name, weight in weights:
                        numel = weight.numel()
                        cpu_buffer[current_offset : current_offset + numel].copy_(
                            weight.flatten()
                        )
                        self._weight_metadata[layer_idx][name] = {
                            "dtype": dtype,
                            "phase_id": phase_idx,
                            "resident": resident_phase,
                            "offset": current_offset,
                            "numel": numel,
                            "shape": weight.shape,
                        }

                        if not resident_phase:
                            weight.data = torch.empty(
                                (1,), device=self.device, dtype=dtype
                            )
                        current_offset += numel

                    self._consolidated_cpu_weights[layer_idx][phase_idx][
                        dtype
                    ] = cpu_buffer

            with self._state_lock:
                if self._layer_complete_locked(layer_idx):
                    self._gpu_layers.add(layer_idx)

        # Warm up initial prefetch window synchronously for first step.
        self.prepare_for_next_req(non_blocking=False)

        self.register_forward_hooks()
        logger.info(
            f"LayerwiseOffloadManager initialized with num prefetched layer: {self.prefetch_size}, total num layers: {self.num_layers}"
        )

    def prepare_for_next_req(self, non_blocking: bool = True):
        if not self.enabled:
            return

        if non_blocking:
            queued = False
            with self._state_lock:
                self._anchor_layer = 0
                for i in range(self.prefetch_size):
                    queued = self._schedule_layer_locked(i) or queued
                for j in range(self.prefetch_size, 2 * self.prefetch_size):
                    queued = self._schedule_layer_locked(j % self.num_layers) or queued
            if queued:
                self._worker_wakeup_event.set()
            return

        for i in range(self.prefetch_size):
            self.prefetch_layer(i, non_blocking=False)

        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def get_target_with_name(self, name: str) -> torch.Tensor:
        """Get the target model weight/buffer to be replaced."""
        if name in self._named_parameters:
            target = self._named_parameters[name]
        else:
            target = self._named_buffers[name]
        return target

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> int:
        """Idempotent layer prefetch.

        Returns:
            int: 1 if layer was queued/launched, 0 otherwise.
        """
        if not self.enabled or self.device is None:
            return 0
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return 0
        if layer_idx not in self._consolidated_cpu_weights:
            return 0
        with self._state_lock:
            if self._layer_complete_locked(layer_idx):
                return 0

        if not non_blocking:
            self._prefetch_layer_blocking(layer_idx)
            return 1

        with self._state_lock:
            queued = self._schedule_layer_locked(layer_idx)
        if queued:
            self._worker_wakeup_event.set()
            return 1
        return 0

    @torch.compiler.disable
    def release_layer(self, layer_idx: int) -> None:
        """Release layer weights from GPU while keeping CPU buffers."""
        if not self.enabled or self.device is None:
            return

        if layer_idx <= 0:
            return

        if layer_idx not in self._weight_metadata:
            return

        with self._state_lock:
            for name, meta in self._weight_metadata.get(layer_idx, {}).items():
                if meta.get("resident", False):
                    continue
                target = self.get_target_with_name(name)
                target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])

            self._prefetch_events.pop(layer_idx, None)
            self._phase_events.pop(layer_idx, None)
            self._prefetch_chunk_events.pop(layer_idx, None)
            self._prefetch_gpu_buffers.pop(layer_idx, None)
            self._prefetch_offsets.pop(layer_idx, None)
            self._scheduled_layers.discard(layer_idx)
            if self._urgent_layer == layer_idx:
                self._urgent_layer = None
            if not all(
                self._phase_is_resident(phase_idx)
                for phase_idx in range(len(self.phase_specs))
            ):
                self._gpu_layers.discard(layer_idx)

    @torch.compiler.disable
    def release_all(self) -> None:
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.release_layer(layer_idx)

    @torch.compiler.disable
    def load_all_layers(self) -> None:
        """Load all layers from CPU to GPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in range(self.num_layers):
            if layer_idx not in self._gpu_layers:
                self.prefetch_layer(layer_idx, non_blocking=False)

    @torch.compiler.disable
    def sync_layer_to_cpu(self, layer_idx: int) -> None:
        """Sync a layer's weights from GPU back to CPU."""
        if not self.enabled or layer_idx not in self._gpu_layers:
            return
        if layer_idx not in self._consolidated_cpu_weights:
            return

        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            gpu_weight = target.data.flatten().cpu()

            dtype = meta["dtype"]
            phase_idx = int(meta["phase_id"])
            cpu_buffer = self._consolidated_cpu_weights[layer_idx][phase_idx][dtype]
            offset = meta["offset"]
            numel = meta["numel"]
            cpu_buffer[offset : offset + numel].copy_(gpu_weight)

    @torch.compiler.disable
    def sync_all_layers_to_cpu(self) -> None:
        """Sync all loaded layers' weights from GPU back to CPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.sync_layer_to_cpu(layer_idx)

    @torch.compiler.disable
    def update_cpu_weights(
        self, weight_dict: Dict[str, torch.Tensor]
    ) -> Set[str] | None:
        """Update consolidated CPU buffers with new weights."""
        if not self.enabled:
            return None

        updated_names: Set[str] = set()
        for name, loaded_weight in weight_dict.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None:
                continue
            meta_layer = self._weight_metadata.get(layer_idx)
            if meta_layer is None or name not in meta_layer:
                continue

            meta = meta_layer[name]
            if tuple(meta["shape"]) != tuple(loaded_weight.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected={tuple(meta['shape'])}, "
                    f"loaded={tuple(loaded_weight.shape)}"
                )

            dtype = meta["dtype"]
            phase_idx = int(meta["phase_id"])
            offset = meta["offset"]
            numel = meta["numel"]
            cpu_buffer = self._consolidated_cpu_weights[layer_idx][phase_idx][dtype]
            cpu_buffer[offset : offset + numel].copy_(
                loaded_weight.to(dtype=dtype).flatten()
            )

            with self._state_lock:
                phase_materialized = self._phase_ready_locked(layer_idx, phase_idx)
            if phase_materialized:
                target = self.get_target_with_name(name)
                target.data.copy_(loaded_weight.to(dtype=target.dtype))

            updated_names.add(name)

        return updated_names

    def iter_cpu_weights(self):
        """Yield (name, tensor) pairs from consolidated CPU buffers."""
        for layer_idx in sorted(self._weight_metadata):
            for name, meta in self._weight_metadata[layer_idx].items():
                dtype = meta["dtype"]
                phase_idx = int(meta["phase_id"])
                offset = meta["offset"]
                numel = meta["numel"]
                shape = meta["shape"]
                cpu_buffer = self._consolidated_cpu_weights[layer_idx][phase_idx][
                    dtype
                ]
                yield name, cpu_buffer[offset : offset + numel].reshape(shape)

    def supports_phase_name(self, phase_name: str) -> bool:
        return phase_name in self.phase_name_to_idx

    @torch.compiler.disable
    def ensure_named_phase_ready(self, layer_idx: int, phase_name: str) -> None:
        phase_idx = self.phase_name_to_idx.get(phase_name)
        if phase_idx is None:
            return

        step_idx = self._current_profile_step_idx()
        waited_bytes, total_bytes = self._estimate_waited_prefetch_bytes(
            layer_idx, phase_idx
        )
        self._record_waited_prefetch_bytes(step_idx, waited_bytes, total_bytes)
        wait_start = self._record_prefetch_cp_wait_start()
        self._ensure_phase_ready(layer_idx, target_phase_idx=phase_idx)
        self._record_prefetch_cp_wait_end(step_idx, layer_idx, wait_start)

    def register_forward_hooks(self) -> None:
        if not self.enabled:
            return

        if self.comm_tracker is not None and self.comm_patch_torch_distributed:
            patched = self.comm_tracker.patch_torch_distributed()
            if not patched:
                logger.warning(
                    "comm-aware offload: unable to patch torch.distributed; "
                    "falling back to explicit communication scopes only."
                )

        layers = getattr(self.model, self.layers_attr_str)

        def make_pre_hook(i):
            @torch.compiler.disable
            def hook(module, input):
                if i == 0:
                    self._begin_step_stats()
                    # Keep this async to avoid main-thread prefetch catch-up.
                    self.prepare_for_next_req(non_blocking=True)

                step_idx = self._current_profile_step_idx()
                waited_bytes, total_bytes = self._estimate_waited_prefetch_bytes(
                    i, 0
                )
                self._record_waited_prefetch_bytes(
                    step_idx, waited_bytes, total_bytes
                )
                wait_start = self._record_prefetch_cp_wait_start()

                # Start async prefetch as early as possible in pre-hook.
                self.prefetch_layer(i, non_blocking=True)
                self._schedule_prefetch_window(i)
                self._ensure_phase_ready(i, target_phase_idx=0)
                self._record_prefetch_cp_wait_end(step_idx, i, wait_start)

            return hook

        def make_post_hook(i):
            @torch.compiler.disable
            def hook(module, input, output):
                self.release_layer(i)
                if i == self.num_layers - 1:
                    self._end_step_stats()

            return hook

        self._forward_hooks.clear()
        for i, layer in enumerate(layers):
            pre_hook_handle = layer.register_forward_pre_hook(make_pre_hook(i))
            post_hook_handle = layer.register_forward_hook(make_post_hook(i))
            self._forward_hooks.extend([pre_hook_handle, post_hook_handle])

        self._start_prefetch_worker()

    def remove_forward_hooks(self) -> None:
        """Remove all registered forward hooks."""
        self._stop_prefetch_worker()
        for hook_handle in self._forward_hooks:
            hook_handle.remove()
        self._forward_hooks.clear()
        if self.comm_tracker is not None:
            self.comm_tracker.unpatch_torch_distributed()


class OffloadableDiTMixin:
    """A mixin that registers forward hooks for a DiT to enable layerwise offload."""

    layer_names: List[str]
    layerwise_offload_managers: list[LayerwiseOffloadManager] = []

    def get_offload_phase_specs(
        self, layer_name: str
    ) -> Sequence[PhaseSpec] | None:
        return None

    def _get_layerwise_offload_manager(
        self, layer_name: str
    ) -> LayerwiseOffloadManager | None:
        for manager in self.layerwise_offload_managers or []:
            if manager.layers_attr_str == layer_name and manager.enabled:
                return manager
        return None

    def configure_layerwise_offload(self, server_args: ServerArgs):
        self.layerwise_offload_managers = []

        comm_aware = bool(
            getattr(server_args, "dit_comm_aware_offload", False)
            or _env_bool("SGLANG_DIT_COMM_AWARE_OFFLOAD", False)
        )
        phase_aware = bool(
            comm_aware and _env_bool("SGLANG_DIT_PHASE_AWARE_PREFETCH", False)
        )
        comm_patch_dist = bool(
            getattr(server_args, "dit_comm_aware_patch_torch_dist", False)
            or _env_bool("SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST", False)
        )
        prefetch_chunk_size_mb = int(
            getattr(server_args, "dit_comm_prefetch_chunk_size_mb", 32)
        )
        prefetch_chunk_size_mb = _env_int(
            "SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB",
            prefetch_chunk_size_mb,
        )
        resident_phase_names = _parse_phase_name_csv(
            getattr(server_args, "dit_offload_resident_phases", "")
        )
        resident_phase_names |= _parse_phase_name_csv(
            _env_str("SGLANG_DIT_OFFLOAD_RESIDENT_PHASES", "")
        )
        if resident_phase_names and not phase_aware:
            logger.info(
                "Ignoring resident phase config because SGLANG_DIT_PHASE_AWARE_PREFETCH is disabled."
            )
            resident_phase_names = set()

        for layer_name in self.layer_names:
            module_list = getattr(self, layer_name, None)
            if module_list is None or not isinstance(module_list, torch.nn.ModuleList):
                continue

            num_layers = len(module_list)
            if server_args.dit_offload_prefetch_size < 1.0:
                prefetch_size = 1 + int(
                    round(server_args.dit_offload_prefetch_size * (num_layers - 1))
                )
            else:
                prefetch_size = int(server_args.dit_offload_prefetch_size)

            phase_specs = (
                self.get_offload_phase_specs(layer_name) if phase_aware else None
            )
            manager = LayerwiseOffloadManager(
                model=self,
                layers_attr_str=layer_name,
                num_layers=num_layers,
                enabled=True,
                pin_cpu_memory=server_args.pin_cpu_memory,
                prefetch_size=prefetch_size,
                comm_aware=comm_aware,
                comm_patch_torch_distributed=comm_patch_dist,
                prefetch_chunk_size_mb=prefetch_chunk_size_mb,
                phase_specs=phase_specs,
                resident_phase_names=resident_phase_names,
                submodule_granularity=True,
            )
            self.layerwise_offload_managers.append(manager)

        logger.info(
            f"Enabled layerwise offload for {self.__class__.__name__} on modules: {self.layer_names}"
        )
        if comm_aware:
            logger.info(
                "Communication-aware chunk-wise offload is enabled "
                f"(chunk={prefetch_chunk_size_mb}MB, patch_torch_dist={comm_patch_dist})."
            )
        if phase_aware:
            logger.info("Phase-aware prefetch is enabled.")
        if resident_phase_names:
            logger.info(
                "Layerwise offload resident phases requested: %s",
                sorted(resident_phase_names),
            )

    @property
    def comm_activity_tracker(self) -> CommunicationActivityTracker | None:
        if not self.layerwise_offload_managers:
            return None
        for manager in self.layerwise_offload_managers:
            if manager.enabled and manager.comm_tracker is not None:
                return manager.comm_tracker
        return None

    @contextmanager
    def comm_region(self, tag: str | None = None):
        tracker = self.comm_activity_tracker
        if tracker is None:
            yield
            return
        with tracker.comm_region(tag):
            yield

    def prepare_for_next_req(self):
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            manager.prepare_for_next_req(non_blocking=True)

    def ensure_offload_phase_ready(
        self, layer_name: str, layer_idx: int, phase_name: str
    ) -> None:
        manager = self._get_layerwise_offload_manager(layer_name)
        if manager is None or not manager.supports_phase_name(phase_name):
            return
        manager.ensure_named_phase_ready(layer_idx, phase_name)

    def quiesce_prefetch_for_comm(self) -> None:
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager.enabled:
                manager.quiesce_copy_stream_for_comm()

    def disable_offload(self) -> None:
        """Disable layerwise offload: load all layers to GPU and remove hooks."""
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager.enabled:
                manager.remove_forward_hooks()
                manager.load_all_layers()

    def enable_offload(self) -> None:
        """Re-enable layerwise offload: sync weights to CPU, release layers, and restore hooks."""
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager.enabled:
                manager.sync_all_layers_to_cpu()
                manager.release_all()
                manager.register_forward_hooks()

    def set_offload_profile_step_idx(self, step_idx: int | None) -> None:
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager.enabled:
                manager.set_logical_step_idx(step_idx)

    def collect_offload_profile_metrics(self) -> Dict[str, Any]:
        if not self.layerwise_offload_managers:
            return {}

        merged: Dict[str, List[float]] = {}
        for manager in self.layerwise_offload_managers:
            if not manager.enabled:
                continue
            data = manager.collect_profile_metrics()
            for key, values in data.items():
                if key not in merged:
                    merged[key] = [0.0] * len(values)
                if len(merged[key]) < len(values):
                    merged[key].extend([0.0] * (len(values) - len(merged[key])))
                for idx, value in enumerate(values):
                    merged[key][idx] += value
        return merged


def iter_materialized_weights(module: torch.nn.Module):
    """Yield (name, tensor) pairs with materialized weights, even under offload."""
    offload_managers: list = []
    if isinstance(module, OffloadableDiTMixin) and module.layerwise_offload_managers:
        offload_managers = [m for m in module.layerwise_offload_managers if m.enabled]

    if not offload_managers:
        yield from module.named_parameters()
        return

    offloaded_names: set[str] = set()
    for manager in offload_managers:
        for name, tensor in manager.iter_cpu_weights():
            offloaded_names.add(name)
            yield name, tensor

    for name, param in module.named_parameters():
        if name not in offloaded_names:
            yield name, param
