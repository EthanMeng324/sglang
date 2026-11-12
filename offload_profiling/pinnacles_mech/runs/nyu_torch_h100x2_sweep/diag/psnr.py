#!/usr/bin/env python3
"""PSNR between two generated videos: decode both mp4s with imageio (ffmpeg backend), compute PSNR per frame on
uint8 RGB (peak 255), then average over frames. Identical frames (MSE=0) are reported separately and excluded
from the mean. usage: python psnr.py a.mp4 b.mp4"""
import sys, numpy as np, imageio.v3 as iio
a, b = iio.imread(sys.argv[1]), iio.imread(sys.argv[2])
if a.shape != b.shape: print(f"shape mismatch {a.shape} vs {b.shape}"); sys.exit(2)
ps, same = [], 0
for fa, fb in zip(a, b):
    mse = np.mean((fa.astype(np.float64) - fb.astype(np.float64)) ** 2)
    if mse == 0: same += 1
    else: ps.append(10 * np.log10(255.0 ** 2 / mse))
mean = f"{np.mean(ps):.3f}" if ps else "inf"
print(f"frames={len(a)} shape={a.shape} identical_frames={same} mean_psnr_db={mean} min_psnr_db={min(ps):.3f}" if ps else f"frames={len(a)} shape={a.shape} identical_frames={same} mean_psnr_db=inf")
