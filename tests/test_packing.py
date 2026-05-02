#!/usr/bin/env python3
"""Quick packing test."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from turboquant.packing import pack_indices, unpack_indices, compute_packed_size
from turboquant import TurboQuantMSE

dim = 1024
n = 1000
x = torch.randn(n, dim)

for bits in [1, 2, 3, 4]:
    quantizer = TurboQuantMSE(dim, bits, device="cpu")
    q_packed = quantizer.quantize(x, pack=True)
    q_unpacked = quantizer.quantize(x, pack=False)
    
    original_bytes = n * dim * 4
    packed_shape = q_packed['indices'].shape
    unpacked_shape = q_unpacked['indices'].shape
    
    print(f"\nBits = {bits}:")
    print(f"  Unpacked: {unpacked_shape}, elements: {q_unpacked['indices'].numel()}")
    print(f"  Packed: {packed_shape}, elements: {q_packed['indices'].numel()}")
    print(f"  Packed flag: {q_packed.get('packed', False)}")
    print(f"  Theoretical packed size: {compute_packed_size(unpacked_shape, bits)} elements")
    print(f"  Actual ratio: {original_bytes / q_packed['indices'].numel():.1f}x")
    
    # Test reconstruction
    x_recon = quantizer.dequantize(q_packed)
    mse = ((x - x_recon)**2).mean().item()
    print(f"  MSE: {mse:.6f}")
