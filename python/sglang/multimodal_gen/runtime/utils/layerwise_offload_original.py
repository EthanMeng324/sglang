import os
import re
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@torch.compiler.disable
def _set_current_offload_layer_label(label: str | None) -> None:
    try:
        setattr(get_forward_context(), "offload_layer_label", label)
    except Exception:
        return


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Adapted from skywork AI Infra diffusion optimize
class LayerwiseOffloadManager:
    """A lightweight layerwise CPU offload manager.

    This utility offloads per-layer parameters/buffers from GPU to CPU, and
    supports async H2D prefetch using a dedicated CUDA stream.

    Typical usage:
    - Construct the manager with the target model and the list-like module
      attribute that represents transformer blocks (e.g. ``blocks``).
    - Call :meth:`initialize` once to offload weights and prefetch layer 0.
    - During forward, call :meth:`prefetch_layer` for the next layer and
      :meth:`release_layer` for the finished layer.
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
    ) -> None:
        self.model = model
        self.layers_attr_str = layers_attr_str
        self.num_layers = num_layers
        self.pin_cpu_memory = pin_cpu_memory
        self.prefetch_size = min(max(1, prefetch_size), self.num_layers)
        self.enabled = bool(enabled and torch.cuda.is_available())
        if not self.enabled:
            return
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.copy_stream = torch.cuda.Stream()

        self._layer_name_re = re.compile(
            rf"(^|\.){re.escape(layers_attr_str)}\.(\d+)(\.|$)"
        )

        # layer_idx -> {dtype: consolidated_pinned_cpu_tensor}
        # stores the consolidated weight from a same layer, of same dtype
        self._consolidated_cpu_weights: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        # layer_idx -> {name: {dtype, offset, numel, shape}}
        # stores the offset and numel of each weight from a same layer, of same dtype
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # layer_idx -> total bytes for profiling
        self._layer_total_bytes: Dict[int, int] = {}
        # layer indices that are already in gpu
        self._gpu_layers: Set[int] = set()
        # layer_idx -> torch.cuda.Event for fine-grained sync, to make sure the weight is resident in pre-hook
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        # layer_idx -> [(copy_done_event, copy_nbytes)]
        self._prefetch_copy_events: Dict[int, List[Tuple[torch.cuda.Event, int]]] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}
        # Store forward hooks for removal
        self._forward_hooks: List[Any] = []
        self._profile_step_seq = -1
        self._logical_step_idx: int | None = None
        self._prefetch_cp_wait_pairs: List[
            Tuple[int, int, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._prefetch_wait_byte_records: List[Tuple[int, int, int]] = []
        self._runtime_state_begin: Dict[int, Dict[str, float]] = {}
        self._runtime_state_end: Dict[int, Dict[str, float]] = {}

        self._initialize()

    def _match_layer_idx(self, name: str) -> int | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    @torch.compiler.disable
    def _initialize(self) -> None:
        if not self.enabled:
            return

        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        # 1. collect and group tensors by layer and dtype
        layer_groups: Dict[int, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]] = {}
        all_tensors = chain(self._named_parameters.items(), self._named_buffers.items())
        for name, tensor in all_tensors:
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            layer_groups.setdefault(layer_idx, {}).setdefault(tensor.dtype, []).append(
                (name, tensor)
            )

        # 2. concat and offload (in pinned memory)
        for layer_idx, dtype_to_params in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}

            for dtype, weights in dtype_to_params.items():
                total_numel = sum(t.numel() for _, t in weights)
                self._layer_total_bytes[layer_idx] = self._layer_total_bytes.get(
                    layer_idx, 0
                ) + int(total_numel * weights[0][1].element_size())

                # create concatenated CPU buffer (in pinned memory)
                cpu_buffer = torch.empty(
                    total_numel, dtype=dtype, pin_memory=self.pin_cpu_memory
                )

                # offload weights to the buffer
                current_offset = 0
                for name, weight in weights:
                    numel = weight.numel()
                    cpu_buffer[current_offset : current_offset + numel].copy_(
                        weight.flatten()
                    )
                    self._weight_metadata[layer_idx][name] = {
                        "dtype": dtype,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": weight.shape,
                    }

                    weight.data = torch.empty((1,), device=self.device, dtype=dtype)

                    current_offset += numel

                self._consolidated_cpu_weights[layer_idx][dtype] = cpu_buffer

        # prefetch the first layer for warm-up
        self.prepare_for_next_req(non_blocking=False)
        if torch.cuda.is_available() and _env_bool(
            "SGLANG_OFFLOAD_EMPTY_CACHE_AFTER_INIT", False
        ):
            torch.cuda.empty_cache()

        self.register_forward_hooks()
        logger.info(
            f"LayerwiseOffloadManager initialized with num prefetched layer: {self.prefetch_size}, total num layers: {self.num_layers}"
        )

    def prepare_for_next_req(self, non_blocking=True):
        """
        Prepare for the next round of denoising loop with prefetching the necessary layers
        """
        for i in range(self.prefetch_size):
            self.prefetch_layer(i, non_blocking=non_blocking)
        if not non_blocking and self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def get_target_with_name(self, name: str) -> torch.Tensor:
        """get the target model weight/buffer to be replaced"""
        if name in self._named_parameters:
            target = self._named_parameters[name]
        else:
            target = self._named_buffers[name]
        return target

    def _begin_profile_step(self) -> None:
        self._profile_step_seq += 1
        self._record_runtime_state(
            self._runtime_state_begin, self._current_profile_step_idx()
        )

    @torch.compiler.disable
    def _current_profile_step_idx(self) -> int:
        if self._logical_step_idx is not None:
            return self._logical_step_idx
        return max(0, self._profile_step_seq)

    @torch.compiler.disable
    def set_logical_step_idx(self, step_idx: int | None) -> None:
        self._logical_step_idx = step_idx

    def _snapshot_runtime_state(self) -> Dict[str, float]:
        managed_gpu_bytes = float(
            sum(int(self._layer_total_bytes.get(layer_idx, 0)) for layer_idx in self._gpu_layers)
        )
        materialized_layer_count = float(len(self._gpu_layers))
        warm_layer0_loaded = 1.0 if 0 in self._gpu_layers else 0.0
        return {
            "offload_managed_gpu_bytes": managed_gpu_bytes,
            "offload_prefetch_buffer_bytes": managed_gpu_bytes,
            "offload_resident_phase_bytes": 0.0,
            "offload_materialized_layer_count": materialized_layer_count,
            "offload_materialized_phase_count": materialized_layer_count,
            "offload_gpu_layer_count": materialized_layer_count,
            "offload_warm_layer0_loaded": warm_layer0_loaded,
        }

    def _record_runtime_state(
        self, bucket: Dict[int, Dict[str, float]], step_idx: int
    ) -> None:
        if not self.enabled:
            return
        bucket[step_idx] = self._snapshot_runtime_state()

    @staticmethod
    def _state_records_to_series(
        records: Dict[int, Dict[str, float]]
    ) -> Dict[str, List[float]]:
        if not records:
            return {}
        max_step = max(records.keys())
        metric_names = sorted({name for state in records.values() for name in state})
        series = {name: [0.0] * (max_step + 1) for name in metric_names}
        for step_idx, state in records.items():
            for name, value in state.items():
                series[name][step_idx] = float(value)
        return series

    def collect_runtime_debug(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "layers_attr_str": self.layers_attr_str,
            "num_layers": self.num_layers,
            "prefetch_size": self.prefetch_size,
            "phase_names": ["layer"],
            "step_state_begin": self._state_records_to_series(
                self._runtime_state_begin
            ),
            "step_state_end": self._state_records_to_series(self._runtime_state_end),
            "current_state": self._snapshot_runtime_state(),
        }

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
    def _estimate_waited_prefetch_bytes(self, layer_idx: int) -> Tuple[int, int]:
        total_bytes = int(self._layer_total_bytes.get(layer_idx, 0))
        if total_bytes <= 0:
            return 0, 0

        if layer_idx in self._gpu_layers:
            ready_event = self._prefetch_events.get(layer_idx)
            if ready_event is None or ready_event.query():
                return 0, total_bytes

        copy_events = list(self._prefetch_copy_events.get(layer_idx, []))
        if not copy_events:
            return total_bytes, total_bytes

        completed_bytes = 0
        for event, nbytes in copy_events:
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
                "Original offload profiling captured waited-byte records but no CUDA "
                "event pairs; exporting zero-filled prefetch_critical_path_wait_ms."
            )
        elif event_errors > 0:
            logger.warning(
                "Original offload profiling dropped %d invalid CUDA event pairs.",
                event_errors,
            )

        result = {
            "prefetch_critical_path_wait_ms": wait_ms_list,
            "prefetch_critical_path_wait_layers": wait_layers_list,
            "prefetch_critical_path_waited_bytes": waited_bytes_list,
            "prefetch_critical_path_total_bytes": total_bytes_list,
        }
        for suffix, records in (
            ("begin", self._runtime_state_begin),
            ("end", self._runtime_state_end),
        ):
            for name, values in self._state_records_to_series(records).items():
                result[f"{name}_{suffix}"] = values
        return result

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        """
        idempotent
        """
        if not self.enabled or self.device is None or self.copy_stream is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if layer_idx not in self._consolidated_cpu_weights:
            return
        self.copy_stream.wait_stream(torch.cuda.current_stream())

        # create gpu buffer and load from CPU buffer
        gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
        copy_events: List[Tuple[torch.cuda.Event, int]] = []
        with torch.cuda.stream(self.copy_stream):
            for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
                gpu_buffer = torch.empty(
                    cpu_buffer.shape, dtype=dtype, device=self.device
                )
                push_detail_nvtx = False
                if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
                    torch.cuda.nvtx.range_push("SGL_PREFETCH_H2D")
                    torch.cuda.nvtx.range_push(
                        f"SGL_PREFETCH_H2D_DETAIL:{self.layers_attr_str}.{layer_idx}"
                    )
                    push_detail_nvtx = True
                try:
                    gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
                    copy_done_event = torch.cuda.Event()
                    copy_done_event.record(self.copy_stream)
                finally:
                    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
                        if push_detail_nvtx:
                            torch.cuda.nvtx.range_pop()
                        torch.cuda.nvtx.range_pop()
                gpu_buffers[dtype] = gpu_buffer
                copy_events.append(
                    (copy_done_event, int(cpu_buffer.numel() * cpu_buffer.element_size()))
                )

        # record the prefetch event of this layer
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        self._prefetch_events[layer_idx] = event
        self._prefetch_copy_events[layer_idx] = copy_events

        # restore model's weights by their metadata using gpu buffer
        for name, meta in self._weight_metadata[layer_idx].items():
            dtype = meta["dtype"]
            gpu_buffer = gpu_buffers[dtype]

            # map the parameter's data to the correct slice of the GPU buffer
            target = self.get_target_with_name(name)
            target.data = gpu_buffer[
                meta["offset"] : meta["offset"] + meta["numel"]
            ].view(meta["shape"])

        self._gpu_layers.add(layer_idx)

    @torch.compiler.disable
    def quiesce_copy_stream_for_comm(self) -> None:
        if not self.enabled or self.copy_stream is None:
            return
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    @torch.compiler.disable
    def release_layer(self, layer_idx: int) -> None:
        """
        lightweight release layer weights
        Basically set the reference count to the gpu weight tensor to zero. The weights on cpu is untouched
        """
        if not self.enabled or self.device is None:
            return

        # clear prefetch event, since it's useless and needs to be reset
        self._prefetch_events.pop(layer_idx, None)
        self._prefetch_copy_events.pop(layer_idx, None)

        if layer_idx <= 0:
            return

        if layer_idx not in self._gpu_layers:
            return

        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])

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

        # Collect current GPU weights and write back to CPU buffer
        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            gpu_weight = target.data.flatten().cpu()

            dtype = meta["dtype"]
            cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
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
        """Update consolidated CPU buffers with new weights.

        When layerwise offload (--dit-layerwise-offload) is enabled, the
        offload manager replaces GPU parameters with small torch.empty((1,))
        placeholders while real weights live in consolidated pinned CPU
        buffers.

        The refit process writes new weights directly into the CPU buffers,
        bypassing the placeholders.  For any layer that happens to be resident
        on the GPU at update time, the live GPU tensor is also updated.

        Args:
            weight_dict: Mapping of parameter name to new weight tensor.

        Returns:
            Set of parameter names that were successfully updated.

        Raises:
            ValueError: If a weight's shape does not match the recorded
                metadata (i.e., the real shape, not the placeholder shape).
        """
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
            offset = meta["offset"]
            numel = meta["numel"]
            cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
            cpu_buffer[offset : offset + numel].copy_(
                loaded_weight.to(dtype=dtype).flatten()
            )

            # If this layer is currently on GPU, update the live parameter.
            if layer_idx in self._gpu_layers:
                target = self.get_target_with_name(name)
                target.data.copy_(loaded_weight.to(dtype=target.dtype))

            updated_names.add(name)

        return updated_names

    def iter_cpu_weights(self):
        """Yield (name, tensor) pairs from consolidated CPU buffers.

        This reconstructs the original weight tensors (with correct shapes)
        from the flat CPU buffers using stored metadata.  Unlike
        model.named_parameters(), which returns (1,) placeholders
        when offload is enabled, this method returns the real weights and
        can be used for checksum computation.
        """
        for layer_idx in sorted(self._weight_metadata):
            for name, meta in self._weight_metadata[layer_idx].items():
                dtype = meta["dtype"]
                offset = meta["offset"]
                numel = meta["numel"]
                shape = meta["shape"]
                cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
                yield name, cpu_buffer[offset : offset + numel].reshape(shape)

    def register_forward_hooks(self) -> None:
        if not self.enabled:
            return

        layers = getattr(self.model, self.layers_attr_str)

        def make_pre_hook(i):
            @torch.compiler.disable
            def hook(module, input):
                _set_current_offload_layer_label(f"{self.layers_attr_str}.{i}")
                if i == 0:
                    self._begin_profile_step()
                step_idx = self._current_profile_step_idx()
                waited_bytes, total_bytes = self._estimate_waited_prefetch_bytes(i)
                self._record_waited_prefetch_bytes(
                    step_idx, waited_bytes, total_bytes
                )
                wait_start = self._record_prefetch_cp_wait_start()

                # wait only for the current layer if it's being prefetched
                if i == 0:
                    self.prepare_for_next_req(non_blocking=False)
                if i in self._prefetch_events:
                    event = self._prefetch_events[i]
                    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
                        # Old implementation's "sync-catchup" equivalent:
                        # event not ready at layer entry means deadline miss.
                        if event.query():
                            torch.cuda.nvtx.range_push("SGL_PREFETCH_OLD_READY_HIT")
                            try:
                                torch.cuda.current_stream().wait_event(event)
                            finally:
                                torch.cuda.nvtx.range_pop()
                        else:
                            torch.cuda.nvtx.range_push(
                                "SGL_PREFETCH_OLD_DEADLINE_MISS_WAIT"
                            )
                            try:
                                torch.cuda.current_stream().wait_event(event)
                            finally:
                                torch.cuda.nvtx.range_pop()
                    else:
                        torch.cuda.current_stream().wait_event(event)

                self._record_prefetch_cp_wait_end(step_idx, i, wait_start)

                # trigger batch prefetch (i + prefetch_size ~ i + 2 * prefetch_size) if needed
                if i % self.prefetch_size == 0:
                    for j in range(i + self.prefetch_size, i + 2 * self.prefetch_size):
                        layer_to_prefetch = j % self.num_layers
                        self.prefetch_layer(layer_to_prefetch, non_blocking=True)

            return hook

        def make_post_hook(i):
            @torch.compiler.disable
            def hook(module, input, output):
                # previous, we wait here, until the copy stream for next layer is finished,
                # now with any prefetch_size, only wait for the copy stream, when the copy stream is for the next layer
                self.release_layer(i)
                _set_current_offload_layer_label(None)
                if i == self.num_layers - 1:
                    self._record_runtime_state(
                        self._runtime_state_end, self._current_profile_step_idx()
                    )

            return hook

        # register prefetch & release hooks for each layer
        self._forward_hooks.clear()
        for i, layer in enumerate(layers):
            pre_hook_handle = layer.register_forward_pre_hook(make_pre_hook(i))
            post_hook_handle = layer.register_forward_hook(make_post_hook(i))
            self._forward_hooks.extend([pre_hook_handle, post_hook_handle])

    def remove_forward_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook_handle in self._forward_hooks:
            hook_handle.remove()
        self._forward_hooks.clear()


class OffloadableDiTMixin:
    """
    A mixin that registers forward hooks for a DiT to enable layerwise offload
    """

    # the list of names of a DiT's layers/blocks
    layer_names: List[str]
    layerwise_offload_managers: list[LayerwiseOffloadManager] = []

    def get_offload_phase_specs(self, layer_name: str):
        return None

    def configure_layerwise_offload(self, server_args: ServerArgs):
        self.layerwise_offload_managers = []
        for layer_name in self.layer_names:
            # a manager per layer-list
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

            manager = LayerwiseOffloadManager(
                model=self,
                layers_attr_str=layer_name,
                num_layers=num_layers,
                enabled=True,
                pin_cpu_memory=server_args.pin_cpu_memory,
                prefetch_size=prefetch_size,
            )
            self.layerwise_offload_managers.append(manager)

        logger.info(
            f"Enabled layerwise offload for {self.__class__.__name__} on modules: {self.layer_names}"
        )

    def prepare_for_next_req(self):
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            manager.prepare_for_next_req(non_blocking=True)

    def ensure_offload_phase_ready(
        self, layer_name: str, layer_idx: int, phase_name: str
    ) -> None:
        return

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

    def collect_offload_runtime_debug(self) -> Dict[str, Any]:
        if not self.layerwise_offload_managers:
            return {}

        managers = []
        for manager in self.layerwise_offload_managers:
            if not manager.enabled:
                continue
            data = manager.collect_runtime_debug()
            if data:
                managers.append(data)
        if not managers:
            return {}
        return {"managers": managers}


def iter_materialized_weights(module: torch.nn.Module):
    """Yield (name, tensor) pairs with materialized weights, even under offload.

    When layerwise offload is active, module.named_parameters() returns
    (1,) placeholders for offloaded layers.  This function reads the
    actual data from the offload manager's CPU buffers and chains it with
    the non-offloaded parameters.
    """
    offload_managers: list = []
    if isinstance(module, OffloadableDiTMixin) and module.layerwise_offload_managers:
        offload_managers = [m for m in module.layerwise_offload_managers if m.enabled]

    if not offload_managers:
        yield from module.named_parameters()
        return

    # Collect offloaded names and their real tensors from CPU buffers.
    offloaded_names: set[str] = set()
    for manager in offload_managers:
        for name, tensor in manager.iter_cpu_weights():
            offloaded_names.add(name)
            yield name, tensor

    # Yield non-offloaded parameters (e.g. final norms, embeddings).
    for name, param in module.named_parameters():
        if name not in offloaded_names:
            yield name, param
