#!/usr/bin/env python3
"""Feasibility test for a GPU-side prefetch gate built on CUDA stream memory operations.

Compute stream A:  [compute kernel] [cuStreamWriteValue32 flag=1] [window kernel ~1.2 ms] [write flag=0] ...
Copy stream B (worker thread): per chunk  [cuStreamWaitValue32 flag==0] [memcpy H2D chunk] ...

Checks: (1) the driver calls work from torch streams / a worker thread, (2) no chunk *starts* inside a window,
(3) the leak per window is at most the chunk in flight, (4) DMA stays back-to-back between windows for
small chunks (async issue, no per-chunk host sync), (5) back-to-back windows (no compute between them).
Single GPU, no NCCL."""
import ctypes, os, sys, threading, time
from ctypes import POINTER, c_int, c_uint, c_uint32, c_uint64, c_void_p, byref
import torch

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
torch.cuda.init()
torch.ones(1, device=dev)  # make the primary context current in this thread


def find_libcuda() -> str:
    with open("/proc/self/maps") as f:
        for ln in f:
            if "libcuda.so" in ln:
                return ln.split()[-1]
    return "libcuda.so.1"


cuda = ctypes.CDLL(find_libcuda())


def fn(names, argtypes):
    for n in names:
        f = getattr(cuda, n, None)
        if f is not None:
            f.restype = c_int
            f.argtypes = argtypes
            return f, n
    raise RuntimeError(f"none of {names} in libcuda")


cuStreamWriteValue32, wname = fn(["cuStreamWriteValue32_v2", "cuStreamWriteValue32"], [c_void_p, c_uint64, c_uint32, c_uint])
cuStreamWaitValue32, waname = fn(["cuStreamWaitValue32_v2", "cuStreamWaitValue32"], [c_void_p, c_uint64, c_uint32, c_uint])
cuMemAlloc, _ = fn(["cuMemAlloc_v2", "cuMemAlloc"], [POINTER(c_uint64), ctypes.c_size_t])
cuMemsetD32, _ = fn(["cuMemsetD32_v2", "cuMemsetD32"], [c_uint64, c_uint32, ctypes.c_size_t])
cuDeviceGetAttribute, _ = fn(["cuDeviceGetAttribute"], [POINTER(c_int), c_int, c_int])
cuDevicePrimaryCtxRetain, _ = fn(["cuDevicePrimaryCtxRetain"], [POINTER(c_void_p), c_int])
cuCtxSetCurrent, _ = fn(["cuCtxSetCurrent"], [c_void_p])
CU_STREAM_WAIT_VALUE_EQ = 0x1
print(f"libcuda: {find_libcuda()}  write={wname} wait={waname}")
for attr, name in ((92, "CAN_USE_STREAM_MEM_OPS_V1"), (93, "CAN_USE_64_BIT_STREAM_MEM_OPS_V1"), (94, "CAN_USE_STREAM_WAIT_VALUE_NOR_V1")):
    v = c_int(-1); rc = cuDeviceGetAttribute(byref(v), attr, 0); print(f"  attr {attr} {name}: rc={rc} value={v.value}")

# flag memory via cuMemAlloc (independent of the caching allocator)
flag_ptr = c_uint64(0)
rc = cuMemAlloc(byref(flag_ptr), 4096); assert rc == 0, f"cuMemAlloc rc={rc}"
rc = cuMemsetD32(flag_ptr, 0, 1024); assert rc == 0, f"cuMemsetD32 rc={rc}"
FLAG = flag_ptr.value
# and one inside a torch tensor (caching allocator) to see whether that works too
flag_t = torch.zeros(1024, dtype=torch.int32, device=dev)
FLAG_T = flag_t.data_ptr()

A = torch.cuda.Stream(device=dev)   # "compute"
B = torch.cuda.Stream(device=dev)   # "copy"
x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
y = torch.empty_like(x)
# size the kernels: find a matmul repeat count giving ~1.2 ms (window) and ~2 ms (compute)
def mm(n):
    for _ in range(n):
        torch.mm(x, x, out=y)
torch.cuda.synchronize()
for n in (1, 2, 4, 8):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); mm(n); e1.record(); torch.cuda.synchronize()
    print(f"  mm x{n}: {e0.elapsed_time(e1):.3f} ms")
e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
e0.record(); mm(1); e1.record(); torch.cuda.synchronize(); per = e0.elapsed_time(e1)
n_win = max(1, round(1.2 / per)); n_comp = max(1, round(2.0 / per))
print(f"  per mm {per:.3f} ms -> window = {n_win} mm, compute = {n_comp} mm")

host = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
dst = torch.empty_like(host, device=dev)


def write_flag(stream, value, ptr=None):
    rc = cuStreamWriteValue32(c_void_p(stream.cuda_stream), c_uint64(ptr or FLAG), c_uint32(value), c_uint(0))
    if rc != 0:
        raise RuntimeError(f"cuStreamWriteValue32 rc={rc}")


def wait_flag(stream, value, ptr=None):
    rc = cuStreamWaitValue32(c_void_p(stream.cuda_stream), c_uint64(ptr or FLAG), c_uint32(value), c_uint(CU_STREAM_WAIT_VALUE_EQ))
    if rc != 0:
        raise RuntimeError(f"cuStreamWaitValue32 rc={rc}")


def run_case(name, *, n_windows, chunk_mb, gated, b2b=False, total_mb=2048, flag_ptr=None, thread=True):
    """Returns summary dict."""
    base = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(A):
        base.record(A)
    torch.cuda.synchronize()
    win = []  # (start_ev, end_ev)
    chunks = []  # (start_ev, end_ev, nbytes)
    chunk = chunk_mb * 1024 * 1024
    n_chunks = total_mb * 1024 * 1024 // chunk
    err = []

    def copy_worker():
        try:
            torch.cuda.set_device(dev)
            torch.ones(1, device=dev)  # context current in this thread
            off = 0
            for i in range(n_chunks):
                if gated:
                    wait_flag(B, 0, flag_ptr)
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(B)
                with torch.cuda.stream(B):
                    dst[off:off + chunk].copy_(host[off:off + chunk], non_blocking=True)
                e.record(B)
                chunks.append((s, e, chunk))
                off += chunk
                if off + chunk > host.numel():
                    off = 0
        except Exception as ex:  # noqa
            err.append(repr(ex))

    t = threading.Thread(target=copy_worker, daemon=True)
    t0 = time.perf_counter()
    if thread:
        t.start()
    else:
        copy_worker()
    with torch.cuda.stream(A):
        for i in range(n_windows):
            if not b2b:
                mm(n_comp)
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            write_flag(A, 1, flag_ptr)
            s.record(A)
            mm(n_win)
            e.record(A)
            write_flag(A, 0, flag_ptr)
            win.append((s, e))
    t.join()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    if err:
        print(f"[{name}] worker error: {err}")
        return None
    W = [(base.elapsed_time(s), base.elapsed_time(e)) for s, e in win]
    C = [(base.elapsed_time(s), base.elapsed_time(e), nb) for s, e, nb in chunks]
    # overlap analysis
    started_inside = 0; leak_ms = 0.0; leak_bytes = 0.0; windows_with_leak = 0
    wi = 0
    for ws, we in W:
        leaked_here = False
        for cs, ce, nb in C:
            if ce <= ws or cs >= we:
                continue
            ov = min(ce, we) - max(cs, ws)
            leak_ms += ov; leak_bytes += nb * ov / max(ce - cs, 1e-6)
            leaked_here = True
            if cs >= ws:
                started_inside += 1
        if leaked_here:
            windows_with_leak += 1
    win_ms = sum(we - ws for ws, we in W)
    copy_ms = sum(ce - cs for cs, ce, _ in C)
    span = (max(ce for _, ce, _ in C) - min(cs for cs, _, _ in C)) if C else 0
    gb = sum(nb for *_, nb in C) / 1e9
    # DMA duty outside windows: copy time / (span - window time within span)
    out_ms = span - sum(min(we, C[-1][1]) - max(ws, C[0][0]) for ws, we in W if we > C[0][0] and ws < C[-1][1])
    print(f"[{name}] chunk={chunk_mb}MB gated={gated} b2b={b2b} windows={n_windows} ({win_ms:.0f} ms) copies={len(C)} {gb:.2f} GB in {copy_ms:.0f} ms busy / span {span:.0f} ms"
          f" -> busy BW {gb/copy_ms*1e3:.1f} GB/s, span BW {gb/span*1e3:.1f} GB/s, outside-window duty {copy_ms - leak_ms:.0f}/{out_ms:.0f} = {(copy_ms-leak_ms)/max(out_ms,1e-6)*100:.0f}%"
          f" | leak: {windows_with_leak}/{len(W)} windows, {leak_ms:.1f} ms total ({leak_ms/max(len(W),1)*1e3:.0f} us/window), {leak_bytes/1e6:.0f} MB; chunks STARTED inside a window: {started_inside}; wall {wall*1e3:.0f} ms")
    return dict(busy_bw=gb / copy_ms * 1e3, leak_ms=leak_ms, started_inside=started_inside, windows=len(W))


print("\n== 1. ungated baseline (chunks overlap windows freely) ==")
run_case("ungated-4MB", n_windows=200, chunk_mb=4, gated=False)
print("\n== 2. gated, chunk sizes ==")
for mb in (16, 4, 2, 1):
    run_case(f"gated-{mb}MB", n_windows=200, chunk_mb=mb, gated=True)
print("\n== 3. gated with the flag inside a torch tensor (caching allocator) ==")
run_case("gated-4MB-torchflag", n_windows=100, chunk_mb=4, gated=True, flag_ptr=FLAG_T)
print("\n== 4. back-to-back windows (no compute between) ==")
run_case("gated-4MB-b2b", n_windows=300, chunk_mb=4, gated=True, b2b=True, total_mb=1024)
print("\n== 5. async-issue DMA throughput without windows (chunking tax check) ==")
for mb in (32, 16, 8, 4, 2, 1):
    run_case(f"nowin-{mb}MB", n_windows=0, chunk_mb=mb, gated=True, total_mb=2048)
print("\n== 6. synchronous per-chunk issue (current _copy_one_chunk style) for comparison ==")
for mb in (16, 4):
    base = torch.cuda.Event(enable_timing=True); base.record(B); torch.cuda.synchronize()
    chunk = mb * 1024 * 1024; n = 2048 * 1024 * 1024 // chunk; off = 0
    t0 = time.perf_counter(); s = torch.cuda.Event(enable_timing=True); s.record(B)
    for i in range(n):
        with torch.cuda.stream(B):
            dst[off:off + chunk].copy_(host[off:off + chunk], non_blocking=False)
            ev = torch.cuda.Event(); ev.record(B)
        off = (off + chunk) % (host.numel() - chunk + 1)
    e = torch.cuda.Event(enable_timing=True); e.record(B); torch.cuda.synchronize()
    print(f"[sync-{mb}MB] {n} chunks, 2.15 GB in {s.elapsed_time(e):.0f} ms -> {2.147/s.elapsed_time(e)*1e3:.1f} GB/s (wall {(time.perf_counter()-t0)*1e3:.0f} ms)")
print("\nDONE memops_test")
