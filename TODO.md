# TurboQuant Production Package - Master TODO

**Objective:** Transform TurboQuant implementation into a production-ready, pip-installable package with comprehensive benchmarks proving paper claims.

**Current Status:** Initial implementation exists with MSE variant working, PROD variant partially implemented. Needs completion, packaging, and validation.

---

## PHASE 1: CRITICAL BUG FIXES ⚠️ (Priority: CRITICAL)

### Task 1.1: Fix PROD Dequantization Bug
**File:** `turboquant/turboquant_prod.py:111-115`  
**Issue:** QJL residual reconstruction is `pass` - never implemented  
**Paper Requirement:** `x̃ = x̃_mse + γ·Sᵀ·qjl`  
**Acceptance:** PROD dequantization matches paper Algorithm 2

### Task 1.2: Fix Inner Product Correction Scaling
**File:** `turboquant/turboquant_prod.py:156`  
**Issue:** Magic number `0.1` scaling factor  
**Paper Requirement:** `γ·√(π/2)·||r|| / √d`  
**Acceptance:** Unbiased inner product estimation validated

### Task 1.3: Compute and Store Gamma (Residual Norm)
**File:** `turboquant/turboquant_prod.py`  
**Issue:** `γ = ||r||₂` not computed in quantization  
**Acceptance:** Gamma properly stored and used in dequantization

---

## PHASE 2: IMPLEMENTATION COMPLETION 🔧 (Priority: HIGH)

### Task 2.1: Implement Beta Distribution Centroids
**Current:** Gaussian approximation (`codebook.py`)  
**Required:** Exact Beta((d-1)/2, (d-1)/2) Lloyd-Max centroids  
**Impact:** Better quality at low bit-widths  
**Acceptance:** Distortion within 1.1× of paper bounds

### Task 2.2: Complete vLLM Integration
**File:** `vllm_integration/` placeholder files  
**Issue:** Stores original values instead of quantized  
**Required:** Full KV cache quantization in vLLM  
**Acceptance:** Llama-3.1-8B runs with TurboQuant KV cache

### Task 2.3: Add SRHT (Fast Rotation) Option
**Current:** O(d²) matrix multiplication  
**Required:** Subsampled Randomized Hadamard Transform O(d log d)  
**Impact:** 10-100× speedup for large dimensions  
**Acceptance:** d=4096 rotation < 1ms on GPU

### Task 2.4: Add Non-Integer Bit Width Support
**Paper:** 2.5-bit, 3.5-bit via outlier channel splitting  
**Required:** Split channels by norm, use different precisions  
**Acceptance:** 2.5-bit achieves 6.4× compression with <2% quality drop

---

## PHASE 3: PACKAGE STRUCTURING 📦 (Priority: HIGH)

### Task 3.1: Create Professional Package Layout
```
turboquant/
├── src/turboquant/           # Source code
│   ├── __init__.py
│   ├── core/                 # Core algorithms
│   │   ├── mse.py           # TurboQuantMSE
│   │   ├── prod.py          # TurboQuantPROD
│   │   ├── rotation.py      # Random rotation
│   │   ├── codebook.py      # Lloyd-Max centroids
│   │   └── qjl.py           # Quantized JL
│   ├── integrations/         # Framework integrations
│   │   ├── vllm.py          # vLLM KV cache
│   │   ├── transformers.py  # HuggingFace
│   │   └── pytorch.py       # nn.Module wrappers
│   └── utils/               # Utilities
│       ├── packing.py       # Bit packing
│       └── validation.py    # Input validation
├── tests/                    # Comprehensive tests
├── benchmarks/               # Performance benchmarks
├── docs/                     # Documentation
├── pyproject.toml           # Package metadata
└── README.md
```

### Task 3.2: Create pyproject.toml
- Package metadata (name, version, description)
- Dependencies (torch, numpy, etc.)
- Optional extras ([vllm], [transformers], [benchmarks])
- Entry points for CLI
- Build system configuration

### Task 3.3: Create Setup Files
- `setup.py` for editable installs
- `requirements.txt` with pinned versions
- `requirements-dev.txt` for development
- `MANIFEST.in` for package data

### Task 3.4: Add CI/CD Configuration
- `.github/workflows/tests.yml` - Run tests on PR
- `.github/workflows/benchmarks.yml` - Nightly benchmarks
- `.github/workflows/release.yml` - PyPI publishing
- Pre-commit hooks for code quality

### Task 3.5: Documentation Structure
- README.md with badges and quickstart
- CONTRIBUTING.md for contributors
- CODE_OF_CONDUCT.md
- LICENSE (Apache 2.0 or MIT)
- docs/ with Sphinx configuration

---

## PHASE 4: COMPREHENSIVE BENCHMARKS 📊 (Priority: HIGH)

### Task 4.1: Distortion Rate Validation (TEST-01)
**Validates:** Paper Claim C1 - D_mse ≤ 2.7/4^b  
**Implementation:** `benchmarks/test_01_distortion.py`

```python
# Test for bits=1..8, dimensions=64..2048
# Generate 10,000 random unit vectors per config
# Compute D_mse = E[||x - x̃||²]
# Assert D_mse ≤ 2.7/(4^b) * 1.05 tolerance
```

**Success Criteria:**
| Bits | Max D_mse | Required Pass |
|------|-----------|---------------|
| 1 | 0.68 | ✓ |
| 2 | 0.17 | ✓ |
| 3 | 0.043 | ✓ |
| 4 | 0.011 | ✓ |

### Task 4.2: Inner Product Accuracy (TEST-02)
**Validates:** Paper Claim C2 - Unbiased inner product  
**Implementation:** `benchmarks/test_02_inner_product.py`

```python
# Generate orthogonal vector pairs
# Compute true and quantized inner products
# Assert |bias| < 0.001, correlation > 0.999
```

### Task 4.3: Perplexity Benchmarks (TEST-03)
**Validates:** KV cache quality on language modeling  
**Models:** Llama-3.1-8B, Ministral-7B  
**Datasets:** WikiText-2, C4  
**Implementation:** `benchmarks/test_03_perplexity.py`

| Config | Target ΔPPL | Hard Limit |
|--------|-------------|------------|
| 4.0-bit | ≤ +0.1 | ≤ +0.5 |
| 3.5-bit | ≤ +0.5 | ≤ +1.0 |
| 2.5-bit | ≤ +2.0 | ≤ +3.0 |

### Task 4.4: LongBench Evaluation (TEST-04)
**Validates:** Paper Claim C3 - 3.5-bit matches FP  
**Implementation:** `benchmarks/test_04_longbench.py`

```bash
# Run LongBench-E on Llama-3.1-8B
# Compare 3.5-bit vs FP16
# Assert score ≥ 49.5 (vs FP16=50.06)
```

### Task 4.5: Needle-in-Haystack (TEST-05)
**Validates:** Paper Claim C4 - Extreme context retrieval  
**Contexts:** 4k, 8k, 16k, 32k, 64k, 128k, 256k+  
**Implementation:** `benchmarks/test_05_needle.py`

| Config | Min Score |
|--------|-----------|
| 4-bit | ≥ 0.99 |
| 3.5-bit | ≥ 0.97 |
| 2.5-bit | ≥ 0.75 |

### Task 4.6: Memory Compression (TEST-09)
**Validates:** Paper Claim C5 - 5× compression  
**Implementation:** `benchmarks/test_09_compression.py`

| Bits | Theoretical | Required Actual |
|------|-------------|-----------------|
| 4.0 | 8.0× | ≥ 7.5× |
| 3.5 | 9.14× | ≥ 8.5× |
| 2.5 | 12.8× | ≥ 12.0× |

### Task 4.7: Throughput Benchmarks (TEST-08)
**Validates:** Paper Claim C6 - Zero indexing overhead  
**Implementation:** `benchmarks/test_08_throughput.py`

- Prefill throughput (tokens/sec)
- Decode throughput (tokens/sec)
- End-to-end latency (TTFT, TPOT)
- Assert ≤ 10% degradation vs FP16

---

## PHASE 5: API & DOCUMENTATION 📝 (Priority: MEDIUM)

### Task 5.1: Simple API Implementation
```python
import turboquant as tq

# One-line usage
compressed = tq.quantize(vector, bits=4)
reconstructed = tq.dequantize(compressed)
```

### Task 5.2: Advanced API
```python
from turboquant import TurboQuantConfig, TurboQuantMSE

config = TurboQuantConfig(
    dim=4096,
    bits=3.5,
    use_beta_distribution=True,
    rotation_type="srht",  # or "random"
)
quantizer = TurboQuantMSE(config)
```

### Task 5.3: Integration Guides
- [ ] vLLM integration tutorial
- [ ] HuggingFace Transformers guide
- [ ] PyTorch nn.Module wrapper
- [ ] Vector DB integration (FAISS, Milvus)

### Task 5.4: CLI Tool
```bash
turboquant benchmark --model llama-3.1-8b --bits 3.5
turboquant compress --input vectors.npy --output quantized.tq --bits 4
turboquant verify --paper-claims all
```

---

## PHASE 6: VERIFICATION & VALIDATION ✅ (Priority: MEDIUM)

### Task 6.1: Unit Tests (100% coverage target)
- `tests/test_rotation.py` - Orthogonality, determinism
- `tests/test_codebook.py` - Centroid correctness
- `tests/test_mse.py` - Quantization roundtrip
- `tests/test_prod.py` - Unbiasedness, variance
- `tests/test_packing.py` - Pack/unpack correctness

### Task 6.2: Integration Tests
- vLLM with TurboQuant KV cache
- HuggingFace pipeline with quantized model
- Multi-GPU distributed training

### Task 6.3: Paper Claim Verification Matrix
| Claim | Tests | Status | Evidence |
|-------|-------|--------|----------|
| C1: D_mse ≤ 2.7/4^b | TEST-01 | ⏳ | Pending |
| C2: Unbiased IP | TEST-02 | ⏳ | Pending |
| C3: 3.5-bit = FP | TEST-03,04 | ⏳ | Pending |
| C4: 2.5-bit marginal | TEST-03,04 | ⏳ | Pending |
| C5: 5× compression | TEST-09 | ⏳ | Pending |
| C6: Zero indexing | TEST-08 | ⏳ | Pending |

---

## DEPENDENCIES & ORDER

```
PHASE 1 (Critical Bugs)
    ↓
PHASE 2 (Completion)
    ↓
PHASE 3 (Packaging) ← Can parallel with 4.1-4.3
    ↓
PHASE 4 (Benchmarks)
    ↓
PHASE 5 (API/Docs)
    ↓
PHASE 6 (Verification)
```

**Parallelizable Groups:**
- Tasks 1.1, 1.2, 1.3 (bug fixes)
- Tasks 3.1-3.5 (packaging)
- Tasks 4.1-4.3 (initial benchmarks)

---

## SUCCESS CRITERIA

### Package Quality
- [ ] `pip install turboquant` works
- [ ] All unit tests pass
- [ ] Type hints throughout
- [ ] 90%+ test coverage
- [ ] Documentation hosted on ReadTheDocs

### Performance Validation
- [ ] 6/6 paper claims validated with evidence
- [ ] Benchmark results published in repo
- [ ] Performance matches or exceeds paper

### Adoption Ready
- [ ] vLLM integration merged/working
- [ ] HuggingFace integration documented
- [ ] PyPI package published
- [ ] 100+ GitHub stars target

---

**Next Action:** Pick first task from Phase 1 and start implementation.
