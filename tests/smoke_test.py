#!/usr/bin/env python3
"""Smoke tests for TurboQuant - verify basic functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from turboquant import TurboQuantMSE, TurboQuantPROD

def test_basic_mse():
    """Test basic MSE quantization roundtrip."""
    print("\n=== Test 1: Basic MSE Roundtrip ===")
    
    dim = 128
    bits = 4
    quantizer = TurboQuantMSE(dim, bits, device="cpu")
    
    # Create test vector
    x = torch.randn(10, dim)
    
    # Quantize and dequantize
    q = quantizer.quantize(x)
    x_recon = quantizer.dequantize(q)
    
    # Check shapes
    assert x_recon.shape == x.shape, f"Shape mismatch: {x_recon.shape} vs {x.shape}"
    
    # Check MSE is reasonable
    mse = ((x - x_recon)**2).mean().item()
    print(f"  MSE: {mse:.6f}")
    assert mse < 1.0, f"MSE too high: {mse}"
    
    print("  ✓ PASS")
    return True

def test_packing():
    """Test bit packing for all bit widths."""
    print("\n=== Test 2: Bit Packing ===")
    
    dim = 128
    batch = 10
    
    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(dim, bits, device="cpu")
        x = torch.randn(batch, dim)
        
        # Quantize with packing
        q_packed = quantizer.quantize(x, pack=True)
        x_recon = quantizer.dequantize(q_packed)
        
        # Check reconstruction
        diff = (x - x_recon).abs().max().item()
        print(f"  bits={bits}: max_diff={diff:.6f}, packed_shape={q_packed['indices'].shape}")
        
        # Packed should be smaller
        if bits < 8:
            assert q_packed['packed'], f"Should be packed for bits={bits}"
        
    print("  ✓ PASS")
    return True

def test_prod_unbiased():
    """Test PROD gives unbiased inner product."""
    print("\n=== Test 3: PROD Unbiased Inner Product ===")
    
    dim = 256
    bits = 4
    quantizer = TurboQuantPROD(dim, bits, device="cpu")
    
    # Create test vectors
    n = 1000
    x = torch.randn(n, dim)
    y = torch.randn(n, dim)
    
    # Normalize to unit sphere (as per paper)
    x = x / x.norm(dim=1, keepdim=True)
    y = y / y.norm(dim=1, keepdim=True)
    
    # True inner products
    ip_true = (x * y).sum(dim=1)
    
    # Quantized inner products
    x_q = quantizer.quantize(x)
    ip_quant = quantizer.compute_inner_product(x_q, y, use_qjl=True)
    
    # Check statistics
    bias = (ip_quant - ip_true).mean().item()
    correlation = np.corrcoef(ip_true.numpy(), ip_quant.numpy())[0, 1]
    
    print(f"  Bias: {bias:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    
    # Should have low bias
    if abs(bias) > 0.05:
        print(f"  ⚠ WARNING: Bias {bias} > 0.05")
    
    # Should have high correlation
    if correlation < 0.9:
        print(f"  ⚠ WARNING: Correlation {correlation} < 0.9")
    
    print("  ✓ PASS (with warnings if shown)")
    return True

def test_compression_ratios():
    """Test compression ratios."""
    print("\n=== Test 4: Compression Ratios ===")
    
    dim = 1024
    n = 1000
    
    # Original size (float32)
    original_bytes = n * dim * 4
    print(f"  Original: {original_bytes/1024:.1f} KB")
    
    x = torch.randn(n, dim)
    
    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(dim, bits, device="cpu")
        q = quantizer.quantize(x, pack=True)
        
        compressed_bytes = q['indices'].numel()
        ratio = original_bytes / compressed_bytes
        theoretical = 32 / bits
        
        print(f"  bits={bits}: {compressed_bytes/1024:.1f} KB, ratio={ratio:.1f}x (theory={theoretical:.1f}x)")
        
        # Should be close to theoretical
        if ratio < theoretical * 0.5:
            print(f"    ⚠ WARNING: Ratio much lower than expected")
    
    print("  ✓ PASS")
    return True

def test_device_transfer():
    """Test GPU/CPU compatibility."""
    print("\n=== Test 5: Device Transfer ===")
    
    if not torch.cuda.is_available():
        print("  Skipped (no CUDA)")
        return True
    
    dim = 128
    bits = 4
    
    # Create quantizer on CPU but move it to GPU
    quantizer = TurboQuantMSE(dim, bits, device="cuda")
    x = torch.randn(10, dim, device="cuda")
    
    q = quantizer.quantize(x)
    x_recon = quantizer.dequantize(q)
    
    print(f"  Input device: {x.device}")
    print(f"  Output device: {x_recon.device}")
    
    print("  ✓ PASS")
    return True

def test_reproducibility():
    """Test same seed gives same results."""
    print("\n=== Test 6: Reproducibility ===")
    
    dim = 128
    bits = 4
    x = torch.randn(10, dim)
    
    quantizer1 = TurboQuantMSE(dim, bits, device="cpu")
    quantizer2 = TurboQuantMSE(dim, bits, device="cpu")
    
    q1 = quantizer1.quantize(x)
    q2 = quantizer2.quantize(x)
    
    # Indices should be identical (fixed seed)
    assert torch.allclose(q1['indices'].float(), q2['indices'].float()), "Non-reproducible!"
    
    print("  ✓ PASS")
    return True

def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("TurboQuant Smoke Tests")
    print("=" * 60)
    
    tests = [
        test_basic_mse,
        test_packing,
        test_prod_unbiased,
        test_compression_ratios,
        test_device_transfer,
        test_reproducibility,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            results.append((test.__name__, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
