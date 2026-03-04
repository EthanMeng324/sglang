import os
import re
import threading
from contextlib import contextmanager
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

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
        comm_aware: bool = False,
        comm_patch_torch_distributed: bool = False,
        prefetch_chunk_size_mb: int = 32,
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
        self.enabled = bool(enabled and torch.cuda.is_available())
        if not self.enabled:
            return
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.copy_stream = torch.cuda.Stream()
        self.comm_tracker = CommunicationActivityTracker() if self.comm_aware else None

        self._layer_name_re = re.compile(
            rf"(^|\.){re.escape(layers_attr_str)}\.(\d+)(\.|$)"
        )

        # layer_idx -> {dtype: consolidated_pinned_cpu_tensor}
        # stores the consolidated weight from a same layer, of same dtype
        self._consolidated_cpu_weights: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        # layer_idx -> {name: {dtype, offset, numel, shape}}
        # stores the offset and numel of each weight from a same layer, of same dtype
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # layer indices that are already in gpu
        self._gpu_layers: Set[int] = set()
        # (layer_idx, unit) -> torch.cuda.Event for fine-grained sync
        self._prefetch_events: Dict[Tuple[int, str], torch.cuda.Event] = {}
        # (layer_idx, unit) -> {dtype: next_numel_offset}
        self._unit_copy_offsets: Dict[Tuple[int, str], Dict[torch.dtype, int]] = {}
        # (layer_idx, unit) -> {dtype: gpu_buffer}
        self._unit_gpu_buffers: Dict[
            Tuple[int, str], Dict[torch.dtype, torch.Tensor]
        ] = {}
        # layer_idx -> ordered unit names
        self._layer_units: Dict[int, List[str]] = {}
        # layer_idx -> unit -> {dtype: consolidated_pinned_cpu_tensor}
        self._unit_cpu_weights: Dict[
            int, Dict[str, Dict[torch.dtype, torch.Tensor]]
        ] = {}
        # layer_idx -> unit -> {name: {dtype, offset, numel, shape}}
        self._unit_weight_metadata: Dict[
            int, Dict[str, Dict[str, Dict[str, Any]]]
        ] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}
        # Store forward hooks for removal
        self._forward_hooks: List[Any] = []
        self._state_lock = threading.RLock()
        self._worker_thread: threading.Thread | None = None
        self._worker_stop_event = threading.Event()
        self._worker_wakeup_event = threading.Event()
        self._prefetch_anchor_layer = 0

        self._initialize()

    def _match_layer_idx(self, name: str) -> int | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    def _extract_unit_name(self, name: str, layer_idx: int) -> str:
        if not self.submodule_granularity:
            return "__layer__"
        marker = f"{self.layers_attr_str}.{layer_idx}."
        if marker not in name:
            return "__layer__"
        tail = name.split(marker, 1)[1]
        if not tail:
            return "__layer__"
        head = tail.split(".", 1)[0]
        return head if head else "__layer__"

    def _set_prefetch_anchor(self, layer_idx: int) -> None:
        with self._state_lock:
            self._prefetch_anchor_layer = layer_idx
        self._worker_wakeup_event.set()

    def _prefetch_window_layers(self) -> List[int]:
        with self._state_lock:
            anchor = self._prefetch_anchor_layer
        return [
            j % self.num_layers
            for j in range(anchor + self.prefetch_size, anchor + 2 * self.prefetch_size)
        ]

    def _start_prefetch_worker(self) -> None:
        if not self.comm_aware or not self.enabled:
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
            if self.comm_tracker is not None and self.comm_tracker.is_active():
                self.comm_tracker.wait_until_inactive(
                    stop_event=self._worker_stop_event, timeout=0.05
                )
                continue

            progressed = 0
            for layer_idx in self._prefetch_window_layers():
                if self._worker_stop_event.is_set():
                    return
                if self.comm_tracker is not None and self.comm_tracker.is_active():
                    break
                progressed += self.prefetch_layer(layer_idx, non_blocking=True)

            if progressed > 0:
                # keep running to continuously prefetch until window is complete or paused by comm
                continue

            self._worker_wakeup_event.wait(timeout=0.01)
            self._worker_wakeup_event.clear()

    @torch.compiler.disable
    def _initialize(self) -> None:
        if not self.enabled:
            return

        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        # 1. collect and group tensors by layer -> unit -> dtype
        layer_groups: Dict[
            int, Dict[str, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]]
        ] = {}
        all_tensors = chain(self._named_parameters.items(), self._named_buffers.items())
        for name, tensor in all_tensors:
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            unit = self._extract_unit_name(name, layer_idx)
            layer_groups.setdefault(layer_idx, {}).setdefault(unit, {}).setdefault(
                tensor.dtype, []
            ).append((name, tensor))

        # 2. concat and offload (in pinned memory)
        for layer_idx, units in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}
            self._unit_cpu_weights[layer_idx] = {}
            self._unit_weight_metadata[layer_idx] = {}
            self._layer_units[layer_idx] = sorted(units.keys())

            for unit, dtype_to_params in units.items():
                self._unit_cpu_weights[layer_idx][unit] = {}
                self._unit_weight_metadata[layer_idx][unit] = {}

                for dtype, weights in dtype_to_params.items():
                    total_numel = sum(t.numel() for _, t in weights)

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
                            "unit": unit,
                        }
                        self._unit_weight_metadata[layer_idx][unit][name] = {
                            "dtype": dtype,
                            "offset": current_offset,
                            "numel": numel,
                            "shape": weight.shape,
                        }

                        weight.data = torch.empty((1,), device=self.device, dtype=dtype)
                        current_offset += numel

                    self._unit_cpu_weights[layer_idx][unit][dtype] = cpu_buffer
                    # Keep backward-compatible layer-level buffer for helper APIs.
                    if dtype not in self._consolidated_cpu_weights[layer_idx]:
                        self._consolidated_cpu_weights[layer_idx][dtype] = cpu_buffer

        # prefetch the first layer for warm-up
        self.prepare_for_next_req(non_blocking=False)

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
        self._set_prefetch_anchor(0)

    def get_target_with_name(self, name: str) -> torch.Tensor:
        """get the target model weight/buffer to be replaced"""
        if name in self._named_parameters:
            target = self._named_parameters[name]
        else:
            target = self._named_buffers[name]
        return target

    def _unit_key(self, layer_idx: int, unit: str) -> Tuple[int, str]:
        return (layer_idx, unit)

    def _ensure_unit_copy_state(
        self, layer_idx: int, unit: str
    ) -> Tuple[Dict[torch.dtype, torch.Tensor], Dict[torch.dtype, int]]:
        key = self._unit_key(layer_idx, unit)
        if key not in self._unit_gpu_buffers:
            gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
            offsets: Dict[torch.dtype, int] = {}
            for dtype, cpu_buffer in self._unit_cpu_weights[layer_idx][unit].items():
                gpu_buffers[dtype] = torch.empty(
                    cpu_buffer.shape, dtype=dtype, device=self.device
                )
                offsets[dtype] = 0
            self._unit_gpu_buffers[key] = gpu_buffers
            self._unit_copy_offsets[key] = offsets
        return self._unit_gpu_buffers[key], self._unit_copy_offsets[key]

    @torch.compiler.disable
    def _bind_unit_weights(self, layer_idx: int, unit: str) -> None:
        key = self._unit_key(layer_idx, unit)
        gpu_buffers = self._unit_gpu_buffers[key]
        for name, meta in self._unit_weight_metadata[layer_idx][unit].items():
            dtype = meta["dtype"]
            gpu_buffer = gpu_buffers[dtype]
            target = self.get_target_with_name(name)
            target.data = gpu_buffer[
                meta["offset"] : meta["offset"] + meta["numel"]
            ].view(meta["shape"])

    @torch.compiler.disable
    def _prefetch_unit(
        self,
        layer_idx: int,
        unit: str,
        *,
        non_blocking: bool,
    ) -> int:
        if self.device is None or self.copy_stream is None:
            return 0
        if layer_idx not in self._unit_cpu_weights:
            return 0
        if unit not in self._unit_cpu_weights[layer_idx]:
            return 0

        key = self._unit_key(layer_idx, unit)
        event = self._prefetch_events.get(key)
        if event is not None:
            return 0

        if (
            self.comm_aware
            and non_blocking
            and self.comm_tracker is not None
            and self.comm_tracker.is_active()
        ):
            return 0

        gpu_buffers, offsets = self._ensure_unit_copy_state(layer_idx, unit)
        self.copy_stream.wait_stream(torch.cuda.current_stream())

        launches = 0
        with torch.cuda.stream(self.copy_stream):
            for dtype, cpu_buffer in self._unit_cpu_weights[layer_idx][unit].items():
                if (
                    self.comm_aware
                    and non_blocking
                    and self.comm_tracker is not None
                    and self.comm_tracker.is_active()
                ):
                    break
                gpu_buffer = gpu_buffers[dtype]
                start = offsets[dtype]
                if start >= cpu_buffer.numel():
                    continue

                if self.comm_aware and non_blocking:
                    item_bytes = max(1, cpu_buffer.element_size())
                    chunk_numel = max(1, self.prefetch_chunk_size_bytes // item_bytes)
                    end = min(cpu_buffer.numel(), start + chunk_numel)
                else:
                    end = cpu_buffer.numel()

                gpu_buffer[start:end].copy_(cpu_buffer[start:end], non_blocking=non_blocking)
                offsets[dtype] = end
                launches += 1

        complete = all(
            offsets[dtype] >= self._unit_cpu_weights[layer_idx][unit][dtype].numel()
            for dtype in self._unit_cpu_weights[layer_idx][unit]
        )
        if complete:
            final_event = torch.cuda.Event()
            final_event.record(self.copy_stream)
            self._prefetch_events[key] = final_event
            self._bind_unit_weights(layer_idx, unit)

        return launches

    @torch.compiler.disable
    def _ensure_layer_ready(self, layer_idx: int) -> None:
        with self._state_lock:
            self._ensure_layer_ready_locked(layer_idx)

    @torch.compiler.disable
    def _ensure_layer_ready_locked(self, layer_idx: int) -> None:
        if self.device is None:
            return
        for unit in self._layer_units.get(layer_idx, []):
            key = self._unit_key(layer_idx, unit)
            while key not in self._prefetch_events:
                self._prefetch_unit(
                    layer_idx,
                    unit,
                    non_blocking=False,
                )
            torch.cuda.current_stream().wait_event(self._prefetch_events[key])
        self._gpu_layers.add(layer_idx)

    @torch.compiler.disable
    def prefetch_layer(
        self,
        layer_idx: int,
        non_blocking: bool = True,
    ) -> int:
        """Idempotent layer prefetch. Returns number of launched copy ops."""
        with self._state_lock:
            if not self.enabled or self.device is None or self.copy_stream is None:
                return 0
            if layer_idx < 0 or layer_idx >= self.num_layers:
                return 0
            if layer_idx not in self._layer_units:
                return 0
            if layer_idx in self._gpu_layers:
                return 0

            launches = 0
            for unit in self._layer_units[layer_idx]:
                launches += self._prefetch_unit(
                    layer_idx, unit, non_blocking=non_blocking
                )

            # Mark as resident only if all units were materialized.
            if all(
                self._unit_key(layer_idx, unit) in self._prefetch_events
                for unit in self._layer_units[layer_idx]
            ):
                self._gpu_layers.add(layer_idx)
            return launches

    @torch.compiler.disable
    def release_layer(self, layer_idx: int) -> None:
        """
        lightweight release layer weights
        Basically set the reference count to the gpu weight tensor to zero. The weights on cpu is untouched
        """
        with self._state_lock:
            if not self.enabled or self.device is None:
                return

            if layer_idx <= 0:
                return

            if layer_idx not in self._weight_metadata:
                return

            for name, meta in self._weight_metadata.get(layer_idx, {}).items():
                target = self.get_target_with_name(name)
                target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])

            for unit in self._layer_units.get(layer_idx, []):
                key = self._unit_key(layer_idx, unit)
                self._prefetch_events.pop(key, None)
                self._unit_gpu_buffers.pop(key, None)
                self._unit_copy_offsets.pop(key, None)

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
        if layer_idx not in self._weight_metadata:
            return

        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        # Collect current GPU weights and write back to CPU buffer
        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            gpu_weight = target.data.flatten().cpu()

            dtype = meta["dtype"]
            unit = meta.get("unit", "__layer__")
            cpu_buffer = self._unit_cpu_weights[layer_idx][unit][dtype]
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
            unit = meta.get("unit", "__layer__")
            cpu_buffer = self._unit_cpu_weights[layer_idx][unit][dtype]
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
                unit = meta.get("unit", "__layer__")
                cpu_buffer = self._unit_cpu_weights[layer_idx][unit][dtype]
                yield name, cpu_buffer[offset : offset + numel].reshape(shape)

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
            def hook(module, input):
                if i == 0:
                    self.prepare_for_next_req(non_blocking=False)
                self._ensure_layer_ready(i)
                self._set_prefetch_anchor(i)

            return hook

        def make_post_hook(i):
            def hook(module, input, output):
                self.release_layer(i)

            return hook

        # register prefetch & release hooks for each layer
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
    """
    A mixin that registers forward hooks for a DiT to enable layerwise offload
    """

    # the list of names of a DiT's layers/blocks
    layer_names: List[str]
    layerwise_offload_managers: list[LayerwiseOffloadManager] = []

    def configure_layerwise_offload(self, server_args: ServerArgs):
        self.layerwise_offload_managers = []
        comm_aware = bool(
            getattr(server_args, "dit_comm_aware_offload", False)
            or _env_bool("SGLANG_DIT_COMM_AWARE_OFFLOAD", False)
        )
        comm_patch_dist = bool(
            getattr(server_args, "dit_comm_aware_patch_torch_dist", False)
            or _env_bool("SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST", False)
        )
        prefetch_chunk_size_mb = int(
            getattr(server_args, "dit_comm_prefetch_chunk_size_mb", 32)
        )
        if "SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB" in os.environ:
            prefetch_chunk_size_mb = _env_int(
                "SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB",
                prefetch_chunk_size_mb,
            )
        submodule_granularity = bool(
            getattr(server_args, "dit_comm_prefetch_submodule_granularity", True)
        )
        if "SGLANG_DIT_COMM_PREFETCH_SUBMODULE_GRANULARITY" in os.environ:
            submodule_granularity = _env_bool(
                "SGLANG_DIT_COMM_PREFETCH_SUBMODULE_GRANULARITY",
                submodule_granularity,
            )

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
                comm_aware=comm_aware,
                comm_patch_torch_distributed=comm_patch_dist,
                prefetch_chunk_size_mb=prefetch_chunk_size_mb,
                submodule_granularity=submodule_granularity,
            )
            self.layerwise_offload_managers.append(manager)

        logger.info(
            f"Enabled layerwise offload for {self.__class__.__name__} on modules: {self.layer_names}"
        )
        if comm_aware:
            logger.info(
                "Communication-aware granular offload is enabled "
                f"(chunk={prefetch_chunk_size_mb}MB, "
                f"submodule_granularity={submodule_granularity}, patch_torch_dist={comm_patch_dist})."
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
