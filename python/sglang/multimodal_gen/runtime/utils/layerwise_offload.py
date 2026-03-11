import os


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
        OffloadableDiTMixin,
        iter_materialized_weights,
    )
else:
    from sglang.multimodal_gen.runtime.utils.layerwise_offload_original import (
        LayerwiseOffloadManager,
        OffloadableDiTMixin,
        iter_materialized_weights,
    )


__all__ = [
    "LayerwiseOffloadManager",
    "OffloadableDiTMixin",
    "iter_materialized_weights",
]
