#!/usr/bin/env python3
"""Comprehensive benchmarks for TurboQuant.

Tests all bit widths (1, 2, 3, 4) with:
- MSE quantization (baseline)
- PROD quantization (optimized for inner products)

Produces final results table.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
from turboquant import TurboQuantMSE, TurboQuantPROD

print("=" * 80)
print("TurboQuant Comprehensive Benchmarks")
print("=" * 80)

# Configuration
DIM = 256
NUM_VECTORS = 1000
NUM_TRIALS = 5
BITS_LIST = [1, 2, 3, 4]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
print(f"Dimension: {DIM}")
print(f"Number of vectors: {NUM_VECTORS}")
print(f"Trials per benchmark: {NUM_TRIALS}")

# Generate test data
torch.manual_seed(42)
x_test = torch.randn(NUM_VECTORS, DIM, device=device)
y_test = torch.randn(NUM_VECTORS, DIM, device=device)

# Normalize to unit sphere
x_test = x_test / x_test.norm(dim=1, keepdim=True)
y_test = y_test / y_test.norm(dim=1, keepdim=True)

# True inner products
ip_true = (x_test * y_test).sum(dim=1).cpu().numpy()

def benchmark_mse(bits):
    """Benchmark MSE quantization."""
    quantizer = TurboQuantMSE(DIM, bits, device=device)
    
    # Warmup
    q = quantizer.quantize(x_test[:10])
    _ = quantizer.dequantize(q)
    
    # Timing
    times = []
    for _ in range(NUM_TRIALS):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        q = quantizer.quantize(x_test)
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times) * 1000  # ms
    throughput = NUM_VECTORS / (np.mean(times))
    
    # Accuracy
    q = quantizer.quantize(x_test)
    x_recon = quantizer.dequantize(q)
    mse = ((x_test - x_recon)**2).mean().item()
    
    # Compression ratio
    original_bytes = NUM_VECTORS * DIM * 4
    compressed_bytes = q['indices'].numel()
    compression_ratio = original_bytes / compressed_bytes
    
    return {
        'time_ms': avg_time,
        'throughput': throughput,
        'mse': mse,
        'compression': compression_ratio
    }

def benchmark_prod(bits):
    """Benchmark PROD quantization."""
    quantizer = TurboQuantPROD(DIM, bits, device=device)
    
    # Warmup
    q = quantizer.quantize(x_test[:10])
    _ = quantizer.compute_inner_product(q, y_test[:10], use_qjl=True)
    
    # Timing - quantization
    times_q = []
    for _ in range(NUM_TRIALS):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        q = quantizer.quantize(x_test)
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.perf_counter()
        times_q.append(end - start)
    
    # Timing - inner product
    times_ip = []
    q = quantizer.quantize(x_test)
    for _ in range(NUM_TRIALS):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        ip_est = quantizer.compute_inner_product(q, y_test, use_qjl=True)
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.perf_counter()
        times_ip.append(end - start)
    
    avg_time_q = np.mean(times_q) * 1000
    avg_time_ip = np.mean(times_ip) * 1000
    
    # Accuracy
    mse_results = []
    ip_correlations = []
    
    for trial in range(5):
        torch.manual_seed(trial)
        x = torch.randn(NUM_VECTORS, DIM, device=device)
        y = torch.randn(NUM_VECTORS, DIM, device=device)
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        
        quantizer_trial = TurboQuantPROD(DIM, bits, device=device)
        q = quantizer_trial.quantize(x)
        x_recon = quantizer_trial.dequantize(q)
        
        mse_results.append(((x - x_recon)**2).mean().item())
        
        ip_true_trial = (x * y).sum(dim=1).cpu().numpy()
        ip_est_trial = quantizer_trial.compute_inner_product(q, y, use_qjl=True).cpu().numpy()
        ip_correlations.append(np.corrcoef(ip_true_trial, ip_est_trial)[0, 1])
    
    mse_mean = np.mean(mse_results)
    ip_corr_mean = np.mean(ip_correlations)
    ip_corr_std = np.std(ip_correlations)
    
    # Compression ratio
    q = quantizer.quantize(x_test)
    original_bytes = NUM_VECTORS * DIM * 4
    compressed_bytes = q['indices'].numel() + q['qjl_hash'].numel() // 8  # qjl is 1 bit per hash
    compression_ratio = original_bytes / compressed_bytes
    
    return {
        'time_q_ms': avg_time_q,
        'time_ip_ms': avg_time_ip,
        'mse': mse_mean,
        'ip_corr': ip_corr_mean,
        'ip_corr_std': ip_corr_std,
        'compression': compression_ratio
    }

# Run benchmarks
print("\n" + "=" * 80)
print("Running MSE Benchmarks...")
print("=" * 80)

mse_results = {}
for bits in BITS_LIST:
    print(f"\nBits = {bits}...")
    mse_results[bits] = benchmark_mse(bits)
    r = mse_results[bits]
    print(f"  Time: {r['time_ms']:.2f} ms ({r['throughput']:.0f} vecs/sec)")
    print(f"  MSE: {r['mse']:.6f}")
    print(f"  Compression: {r['compression']:.1f}x")

print("\n" + "=" * 80)
print("Running PROD Benchmarks...")
print("=" * 80)

prod_results = {}
for bits in BITS_LIST:
    print(f"\nBits = {bits}...")
    prod_results[bits] = benchmark_prod(bits)
    r = prod_results[bits]
    print(f"  Quantize: {r['time_q_ms']:.2f} ms")
    print(f"  Inner Product: {r['time_ip_ms']:.2f} ms")
    print(f"  MSE: {r['mse']:.6f}")
    print(f"  IP Correlation: {r['ip_corr']:.4f} ± {r['ip_corr_std']:.4f}")
    print(f"  Compression: {r['compression']:.1f}x")

# Final Summary Table
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print("\n--- TurboQuant MSE ---")
print(f"{'Bits':>6} | {'Time (ms)':>12} | {'Throughput':>12} | {'MSE':>12} | {'Compress':>10}")
print("-" * 60)
for bits in BITS_LIST:
    r = mse_results[bits]
    print(f"{bits:>6} | {r['time_ms']:>12.2f} | {r['throughput']:>11.0f}/s | {r['mse']:>12.6f} | {r['compression']:>9.1f}x")

print("\n--- TurboQuant PROD ---")
print(f"{'Bits':>6} | {'Q Time':>10} | {'IP Time':>10} | {'MSE':>12} | {'IP Corr':>10} | {'Compress':>10}")
print("-" * 70)
for bits in BITS_LIST:
    r = prod_results[bits]
    print(f"{bits:>6} | {r['time_q_ms']:>9.2f}ms | {r['time_ip_ms']:>9.2f}ms | {r['mse']:>12.6f} | {r['ip_corr']:>9.4f} | {r['compression']:>9.1f}x")

# Quality metrics vs paper claims
print("\n" + "=" * 80)
print("Quality Verification vs Paper Claims")
print("=" * 80)

print("\nPaper claims for PROD:")
print("  - Unbiased inner product estimation (E[<y, x_tilde>] = <y, x>)")
print("  - High correlation (>0.99 for b≥3) between quantized and true IPs")

print("\nActual results (averaged over 5 trials with different seeds):")
for bits in BITS_LIST:
    r = prod_results[bits]
    status = "✓ PASS" if r['ip_corr'] > 0.95 else "✗ FAIL" if r['ip_corr'] < 0.9 else "⚠ WARN"
    print(f"  b={bits}: IP correlation = {r['ip_corr']:.4f} ± {r['ip_corr_std']:.4f} {status}")

print("\n" + "=" * 80)
print("Benchmark Complete!")
print("=" * 80)
