"""QJL (Quantized Johnson-Lindenstrauss) for residual quantization.

Implements the QJL transform from TurboQuant paper:
- Quantization: Q_qjl(x) := sign(S . x) where S_ij ~ N(0,1)
- Dequantization: Q_qjl^{-1}(z) := S^T . z (unscaled)
- Scaling: Full reconstruction uses gamma * sqrt(pi/2)/d * S^T . z

The QJL provides 1-bit quantization that enables UNBIASED inner product estimation
when combined with the residual norm gamma.
"""

import torch
import math


class QJL:
    """Quantized Johnson-Lindenstrauss transform.
    
    Provides 1-bit quantization via random projection followed by sign.
    Key property: Enables unbiased inner product estimation.
    """

    def __init__(self, dim: int, num_hashes: int, device="cpu", dtype=torch.float32):
        """Initialize QJL.

        Args:
            dim: Input vector dimension
            num_hashes: Number of hash bits (projection dimension)
            device: Device for computations
            dtype: Data type
        """
        self.dim = dim
        self.num_hashes = num_hashes
        self.device = device
        self.dtype = dtype

        # Generate random Gaussian matrix S (num_hashes x dim)
        # Paper: S_ij ~ N(0,1)
        self.S = torch.randn(num_hashes, dim, device=device, dtype=dtype)
        
        # Normalization: ||S^T S|| should give expected projection norm
        # For computing inner products, we need consistent scaling
        # Store normalization factor
        self.norm_factor = 1.0

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sign(S . x) for input vectors.
        
        This is the QJL quantization: 1 bit per hash.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Binary hash of shape (..., num_hashes) with values in {-1, +1}
        """
        # Compute S @ x^T
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.dim)  # (N, dim)

        # S: (num_hashes, dim), x_flat^T: (dim, N)
        proj = x_flat @ self.S.T  # (N, num_hashes)

        # Sign with {-1, +1}
        h = torch.sign(proj)
        h[h == 0] = 1  # Handle zeros

        return h.reshape(*original_shape, self.num_hashes)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Compute S . x (projection without quantization).
        
        Used for inner product computation: <S.y, sign(S.r)> estimates <y, r>.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Projection of shape (..., num_hashes)
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.dim)
        
        proj = x_flat @ self.S.T  # (N, num_hashes)
        
        return proj.reshape(*original_shape, self.num_hashes)

    def dequantize(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruct from QJL hash: S^T . h.
        
        Note: This returns UNCALED reconstruction. Full reconstruction requires:
        x_tilde = gamma * sqrt(pi/2)/d * S^T . h
        
        where gamma = ||r|| is the residual norm.

        Args:
            h: Hash tensor of shape (..., num_hashes) with values in {-1, +1}

        Returns:
            Reconstructed tensor of shape (..., dim) - UNSCALED
        """
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, self.num_hashes)  # (N, num_hashes)
        
        # S^T . h: (dim, num_hashes) @ (num_hashes, N) -> (dim, N) -> (N, dim)
        x_reconstructed = h_flat @ self.S  # (N, dim)
        
        return x_reconstructed.reshape(*original_shape, self.dim)

    def approximate_inner_product(
        self, h1: torch.Tensor, h2: torch.Tensor
    ) -> torch.Tensor:
        """Approximate inner product using QJL hashes.
        
        For unit vectors, E[h1^T h2 / m] = cos(theta) where theta is angle.
        For general vectors, this estimates <x1, x2> / (||x1|| ||x2||).

        Args:
            h1: Hash of first vector (..., num_hashes)
            h2: Hash of second vector (..., num_hashes)

        Returns:
            Approximate cosine similarity (not scaled by norms)
        """
        # Agreement: h1 == h2 -> +1, h1 != h2 -> -1
        agreement = (h1 == h2).float() * 2 - 1

        # Average over hashes gives cosine similarity estimate
        cos_theta = agreement.mean(dim=-1)

        return cos_theta


def create_qjl(dim: int, num_hashes: int, device="cpu", dtype=torch.float32) -> QJL:
    """Convenience function to create QJL instance.

    Args:
        dim: Input dimension
        num_hashes: Number of hash bits
        device: Device
        dtype: Data type

    Returns:
        QJL instance
    """
    return QJL(dim, num_hashes, device, dtype)
