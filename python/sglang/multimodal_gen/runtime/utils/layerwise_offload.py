import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# Runtime selection between the original layerwise offload and the
# chunk-wise comm-aware implementation.
_USE_CHUNKWISE_IMPL = _env_bool("SGLANG_DIT_COMM_AWARE_OFFLOAD", False)

if _USE_CHUNKWISE_IMPL:
    from sglang.multimodal_gen.runtime.utils.layerwise_offload_chunkwise import (
        LayerwiseOffloadManager,
        OffloadableDiTMixin as _BaseOffloadableDiTMixin,
        PhaseSpec,
        iter_materialized_weights,
    )
else:
    from sglang.multimodal_gen.runtime.utils.layerwise_offload_original import (
        LayerwiseOffloadManager,
        OffloadableDiTMixin as _BaseOffloadableDiTMixin,
        iter_materialized_weights,
    )

    @dataclass(frozen=True)
    class PhaseSpec:
        name: str
        prefixes: tuple[str, ...]


class OffloadableDiTMixin(_BaseOffloadableDiTMixin):
    """Stable offload mixin surface shared by original and chunkwise backends."""

    def get_offload_phase_specs(self, layer_name: str):
        getter = getattr(super(), "get_offload_phase_specs", None)
        if getter is None:
            return None
        return getter(layer_name)

    def ensure_offload_phase_ready(
        self, layer_name: str, layer_idx: int, phase_name: str
    ) -> None:
        ensure = getattr(super(), "ensure_offload_phase_ready", None)
        if ensure is None:
            return
        ensure(layer_name, layer_idx, phase_name)


__all__ = [
    "LayerwiseOffloadManager",
    "OffloadableDiTMixin",
    "PhaseSpec",
    "iter_materialized_weights",
]
