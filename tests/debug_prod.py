#!/usr/bin/env python3
"""Debug script for PROD inner product accuracy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from turboquant import TurboQuantPROD

def debug_prod():
    """Debug PROD implementation step by step."""
    print("=" * 70)
    print("PROD Debugging")
    print("=" * 70)
    
    dim = 256
    bits = 4
    
    # Create simple test case
    torch.manual_seed(42)
    x = torch.randn(100, dim)
    y = torch.randn(100, dim)
    
    # Normalize both to unit sphere
    x = x / x.norm(dim=1, keepdim=True)
    y = y / y.norm(dim=1, keepdim=True)
    
    # True inner products
    ip_true = (x * y).sum(dim=1)
    print(f"\nTrue IP stats: mean={ip_true.mean():.4f}, std={ip_true.std():.4f}")
    
    # Quantize x
    quantizer = TurboQuantPROD(dim, bits, device="cpu")
    x_q = quantizer.quantize(x)
    
    print(f"\nQuantization info:")
    print(f"  MSE bits: {quantizer.mse_bits}")
    print(f"  Total bits: {quantizer.bits}")
    print(f"  QJL scale: {quantizer.qjl_scale:.6f}")
    
    # Get rotated vectors
    x_rot = quantizer.rotator.rotate(x)
    y_rot = quantizer.rotator.rotate(y)
    
    # MSE-only reconstruction
    x_mse_rot = quantizer.codebook.dequantize(x_q['indices'])
    
    print(f"\nMSE reconstruction:")
    print(f"  Original x_rot mean norm: {x_rot.norm(dim=1).mean():.4f}")
    print(f"  MSE x_rot mean norm: {x_mse_rot.norm(dim=1).mean():.4f}")
    
    # Residual
    r_rot = x_rot - x_mse_rot
    gamma = r_rot.norm(dim=1)
    print(f"\nResidual stats:")
    print(f"  Mean gamma (residual norm): {gamma.mean():.4f}")
    print(f"  Mean ||r||/||x||: {(gamma / x_rot.norm(dim=1)).mean():.4f}")
    
    # Test MSE-only inner product
    ip_mse_only = (x_mse_rot * y_rot).sum(dim=1)
    corr_mse = np.corrcoef(ip_true.numpy(), ip_mse_only.numpy())[0, 1]
    print(f"\nMSE-only IP correlation: {corr_mse:.4f}")
    
    # Now test with QJL
    # QJL hash should be sign(S . r)
    r_hash = quantizer.qjl.hash(r_rot)
    
    # Check QJL properties
    print(f"\nQJL hash stats:")
    print(f"  Mean of hash: {r_hash.float().mean():.4f} (should be ~0)")
    print(f"  Fraction +1: {(r_hash > 0).float().mean():.4f}")
    
    # Reconstruct residual using QJL
    r_recon = quantizer.qjl.dequantize(r_hash) * gamma.unsqueeze(1) * quantizer.qjl_scale
    print(f"\nQJL residual reconstruction:")
    print(f"  Original residual norm: {gamma.mean():.4f}")
    print(f"  Reconstructed norm: {r_recon.norm(dim=1).mean():.4f}")
    
    # Check inner product of reconstructed residual
    ip_r = (r_rot * y_rot).sum(dim=1)
    ip_r_recon = (r_recon * y_rot).sum(dim=1)
    corr_r = np.corrcoef(ip_r.numpy(), ip_r_recon.numpy())[0, 1]
    print(f"  Residual IP correlation: {corr_r:.4f}")
    
    # Full reconstruction
    x_full_rot = x_mse_rot + r_recon
    ip_full = (x_full_rot * y_rot).sum(dim=1)
    corr_full = np.corrcoef(ip_true.numpy(), ip_full.numpy())[0, 1]
    print(f"\nFull reconstruction IP correlation: {corr_full:.4f}")
    
    # Now test compute_inner_product method
    ip_method = quantizer.compute_inner_product(x_q, y, use_qjl=True)
    corr_method = np.corrcoef(ip_true.numpy(), ip_method.numpy())[0, 1]
    print(f"compute_inner_product correlation: {corr_method:.4f}")
    
    # Debug the compute_inner_product step by step
    print("\n" + "=" * 70)
    print("Debugging compute_inner_product()")
    print("=" * 70)
    
    y_projected = quantizer.qjl.project(y_rot)
    print(f"y_projected norm: {y_projected.norm(dim=1).mean():.4f}")
    print(f"qjl_hash norm: {x_q['qjl_hash'].norm(dim=1).mean():.4f}")
    
    ip_with_sign = (y_projected * x_q['qjl_hash']).sum(dim=-1)
    print(f"ip_with_sign mean: {ip_with_sign.mean():.4f}")
    print(f"ip_with_sign std: {ip_with_sign.std():.4f}")
    
    # The issue: gamma vs sqrt(dim) scaling
    print(f"\nScaling analysis:")
    print(f"  gamma (residual norm): {x_q['gamma'].squeeze().mean():.4f}")
    print(f"  sqrt(dim): {np.sqrt(dim):.4f}")
    print(f"  qjl_scale: {quantizer.qjl_scale:.6f}")
    
    # Expected vs actual term2
    term2_expected = gamma * (r_rot * y_rot).sum(dim=1) / (r_rot.norm(dim=1) + 1e-8)
    term2_actual = x_q['gamma'].squeeze() * quantizer.qjl_scale * ip_with_sign
    print(f"\nTerm2 comparison:")
    print(f"  Expected residual contribution: {term2_expected.mean():.4f}")
    print(f"  Actual term2: {term2_actual.mean():.4f}")
    print(f"  Ratio: {(term2_actual / (term2_expected + 1e-8)).mean():.4f}")

if __name__ == "__main__":
    debug_prod()
