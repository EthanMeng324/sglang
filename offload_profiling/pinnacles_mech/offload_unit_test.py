#!/usr/bin/env python3
"""Standalone correctness test of the chunk-wise layerwise offload manager (one GPU, no model files).

A stack of synthetic transformer-like blocks (bf16) is run once as a plain module (reference), then
wrapped by LayerwiseOffloadManager exactly as the runtime does (prefetch_size=1, chunked copies,
forward hooks) and run again for several steps. Two checks:
  A. data path: materialize each layer through the manager and compare every bound parameter with
     the original tensor (bitwise);
  B. hooked forward: run the full forward through the hooks and compare with the reference output.
Modes: 'none' (no comm tracker: blocking chunk copies, like launch without collectives) and 'gate'.
usage: offload_unit_test.py [chunk_mb] [mode,...]
"""
import os
import sys
import traceback

os.environ.setdefault("SGLANG_DIT_COMM_AWARE_OFFLOAD", "1")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sglang.multimodal_gen.runtime.utils.comm_aware_prefetch import (  # noqa: E402
    CommunicationActivityTracker,
)
from sglang.multimodal_gen.runtime.utils.layerwise_offload_chunkwise import (  # noqa: E402
    LayerwiseOffloadManager,
)

N_LAYERS, D, TOKENS = 8, 1024, 4096


class Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.l1 = nn.Linear(d, 4 * d)
        self.l2 = nn.Linear(4 * d, d)
        self.scale_shift = nn.Parameter(torch.randn(2, d) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) * (1 + self.scale_shift[0]) + self.scale_shift[1]
        return x + self.l2(torch.nn.functional.gelu(self.l1(h)))


class Model(nn.Module):
    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


def build() -> tuple[Model, torch.Tensor, torch.Tensor, dict]:
    torch.manual_seed(0)
    model = Model(N_LAYERS, D).to("cuda", torch.bfloat16)
    x = torch.randn(2, TOKENS, D, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref = model(x).clone()
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    return model, x, ref, originals


def make_manager(model: Model, mode: str, chunk_mb: int) -> LayerwiseOffloadManager:
    tracker = CommunicationActivityTracker(window_mode=mode) if mode != "none" else None
    return LayerwiseOffloadManager(
        model,
        layers_attr_str="blocks",
        num_layers=N_LAYERS,
        enabled=True,
        pin_cpu_memory=True,
        prefetch_size=1,
        comm_aware=(mode != "none"),
        comm_patch_torch_distributed=False,
        comm_active_window_mode=(mode if mode != "none" else "kernel"),
        comm_tracker=tracker,
        prefetch_chunk_size_mb=chunk_mb,
        phase_aware_mode=False,
    )


def check_data_path(mode: str, chunk_mb: int) -> None:
    model, x, ref, originals = build()
    mgr = make_manager(model, mode, chunk_mb)
    print(f"[{mode} {chunk_mb}MB] deferred_release={mgr._use_deferred_buffer_release} "
          f"phases={len(mgr.phase_specs)} gate_mode={getattr(mgr, '_gate_mode', None)}")
    bad = 0
    entry = mgr._initial_entry_target_phase_idx()
    for i in range(N_LAYERS):
        mgr._ensure_phase_ready(i, target_phase_idx=entry)
        torch.cuda.synchronize()
        for name, p in model.blocks[i].named_parameters():
            full = f"blocks.{i}.{name}"
            o = originals[full]
            if p.data.shape != o.shape or not torch.equal(p.data, o):
                bad += 1
                diff = (p.data.float() - o.float()).abs().max().item() if p.data.shape == o.shape else float("nan")
                print(f"  MISMATCH {full}: shape {tuple(p.data.shape)} vs {tuple(o.shape)}, max|diff|={diff}, "
                      f"ptr=0x{p.data.data_ptr():x} numel={p.data.numel()}")
        mgr.release_layer(i)
    print(f"[{mode} {chunk_mb}MB] A. data path: {'OK' if bad == 0 else f'{bad} parameters wrong'}")
    mgr.remove_forward_hooks()


def check_forward(mode: str, chunk_mb: int, steps: int = 3) -> None:
    model, x, ref, originals = build()
    mgr = make_manager(model, mode, chunk_mb)
    diffs = []
    with torch.no_grad():
        for _ in range(steps):
            out = model(x)
            torch.cuda.synchronize()
            diffs.append((out.float() - ref.float()).abs().max().item())
    finite = all(torch.isfinite(torch.tensor(d)) for d in diffs)
    print(f"[{mode} {chunk_mb}MB] B. hooked forward max|out-ref| per step: {diffs} "
          f"-> {'OK' if finite and max(diffs) == 0.0 else 'WRONG'}")
    mgr.remove_forward_hooks()


if __name__ == "__main__":
    chunk_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["none", "gate"]
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}; chunk={chunk_mb}MB modes={modes}")
    for mode in modes:
        for fn in (check_data_path, check_forward):
            try:
                fn(mode, chunk_mb)
            except Exception:  # noqa: BLE001
                print(f"[{mode} {chunk_mb}MB] {fn.__name__} raised:")
                traceback.print_exc()
            torch.cuda.empty_cache()
