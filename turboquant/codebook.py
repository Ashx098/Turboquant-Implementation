"""Precomputed codebooks for TurboQuant with Beta distribution support.

Implements both Gaussian and Beta distribution Lloyd-Max centroids.
Paper uses Beta((d-1)/2, (d-1)/2) for rotated coordinates.
For high dimensions, Gaussian N(0, 1/d) is a valid approximation.
"""

import torch
import numpy as np

# Optional: Beta distribution support requires scipy
try:
    from scipy.special import betaincinv, gamma as gamma_func
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def generate_centroids_beta(dim: int, bits: int, num_points: int = 50000) -> torch.Tensor:
    """Generate Lloyd-Max centroids for Beta((d-1)/2, (d-1)/2) distribution.
    
    This is the theoretically correct distribution for coordinates after
    random rotation (paper Lemma 1).
    
    PDF: f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
    
    Args:
        dim: Vector dimension (determines Beta parameters)
        bits: Number of bits per coordinate (1, 2, 3, 4)
        num_points: Number of points for numerical integration
        
    Returns:
        Centroids tensor of shape (num_centroids,)
    """
    if not HAS_SCIPY:
        raise ImportError("Beta distribution centroids require scipy. "
                         "Install with: pip install scipy")
    
    alpha = beta = (dim - 1) / 2
    num_centroids = 2 ** bits
    
    # Generate points in [-1, 1] according to Beta distribution
    # Transform: x = 2*u - 1 where u ~ Beta(alpha, beta)
    u = np.linspace(0, 1, num_points)
    
    # Compute CDF values for inverse transform
    # We'll use scipy's betaincinv for the inverse CDF
    cdf_points = np.linspace(0, 1, num_points)
    
    # Lloyd-Max iteration
    # Start with uniform centroids in transformed space
    centroids_u = np.linspace(0, 1, num_centroids + 2)[1:-1]  # Exclude boundaries
    
    # Map to x space
    centroids_x = 2 * centroids_u - 1
    
    # Lloyd-Max iterations
    for iteration in range(100):
        # Compute boundaries (midpoints between centroids)
        boundaries = np.concatenate([
            [-1.0],
            (centroids_x[:-1] + centroids_x[1:]) / 2,
            [1.0]
        ])
        
        # Update centroids to conditional mean in each region
        new_centroids = []
        for i in range(num_centroids):
            a, b = boundaries[i], boundaries[i + 1]
            
            # Sample points in this region
            mask = (u >= (a + 1) / 2) & (u <= (b + 1) / 2)
            if mask.sum() == 0:
                new_centroids.append(centroids_x[i])
                continue
                
            # Conditional mean in x space
            u_region = u[mask]
            x_region = 2 * u_region - 1
            
            # Weight by PDF (Beta PDF in x space)
            pdf_vals = ((1 - x_region**2) ** ((dim - 3) / 2))
            pdf_vals = pdf_vals / pdf_vals.sum()
            
            new_centroid = (x_region * pdf_vals).sum()
            new_centroids.append(new_centroid)
        
        new_centroids = np.array(new_centroids)
        
        # Check convergence
        if np.abs(new_centroids - centroids_x).max() < 1e-6:
            break
            
        centroids_x = new_centroids
    
    return torch.tensor(centroids_x, dtype=torch.float32)


def generate_centroids_gaussian(
    dim: int, bits: int, device="cpu", dtype=torch.float32, unit_variance: bool = True
) -> torch.Tensor:
    """Generate centroids for Gaussian distribution (high-dim approximation).
    
    For large d, Beta((d-1)/2, (d-1)/2) ≈ N(0, 1/d) by CLT.
    This is faster and works well for d >= 64.

    Args:
        dim: Original vector dimension (used for rotation)
        bits: Number of bits per coordinate (1, 2, 3, 4)
        device: Device to create centroids on
        dtype: Data type
        unit_variance: If True, assume unit variance input (default).
                      If False, assume variance = 1/dim.

    Returns:
        Centroids tensor of shape (num_centroids,)
    """
    num_centroids = 2**bits

    # For unit variance input, std = 1
    # For variance = 1/dim, std = 1/sqrt(dim)
    std = 1.0 if unit_variance else 1.0 / (dim**0.5)

    # Generate centroids for uniform Lloyd-Max quantization
    # For Gaussian N(0, sigma^2), use Lloyd-Max centroids

    if bits == 1:
        # 2 centroids: -sigma * sqrt(2/pi), +sigma * sqrt(2/pi)
        # This is the optimal 1-bit quantization for Gaussian
        c = (2.0 / 3.14159) ** 0.5 * std
        centroids = torch.tensor([-c, c], device=device, dtype=dtype)

    elif bits == 2:
        # 4 centroids for Gaussian
        # Approximate Lloyd-Max centroids
        levels = torch.tensor([-1.51, -0.45, 0.45, 1.51], device=device, dtype=dtype)
        centroids = levels * std

    elif bits == 3:
        # 8 centroids for Gaussian
        levels = torch.tensor(
            [-2.40, -1.46, -0.84, -0.28, 0.28, 0.84, 1.46, 2.40],
            device=device,
            dtype=dtype,
        )
        centroids = levels * std

    elif bits == 4:
        # 16 centroids for Gaussian
        levels = torch.tensor(
            [
                -3.40,
                -2.60,
                -2.00,
                -1.50,
                -1.10,
                -0.75,
                -0.45,
                -0.15,
                0.15,
                0.45,
                0.75,
                1.10,
                1.50,
                2.00,
                2.60,
                3.40,
            ],
            device=device,
            dtype=dtype,
        )
        centroids = levels * std

    else:
        raise ValueError(f"bits must be 1, 2, 3, or 4, got {bits}")

    return centroids


def get_centroids(
    dim: int, 
    bits: int, 
    device="cpu", 
    dtype=torch.float32, 
    unit_variance: bool = True,
    use_beta: bool = False
) -> torch.Tensor:
    """Get centroids for quantization.
    
    Args:
        dim: Vector dimension
        bits: Number of bits (1-4)
        device: Device
        dtype: Data type
        unit_variance: Whether input has unit variance
        use_beta: If True, use exact Beta distribution (slow but accurate).
                  If False, use Gaussian approximation (fast, good for d>=64).
    
    Returns:
        Centroids tensor
    """
    if use_beta:
        if not HAS_SCIPY:
            raise ImportError("Beta distribution requires scipy. "
                             "Install with: pip install scipy")
        # Use exact Beta distribution for small dimensions
        centroids = generate_centroids_beta(dim, bits)
        return centroids.to(device=device, dtype=dtype)
    else:
        # Use Gaussian approximation
        return generate_centroids_gaussian(dim, bits, device, dtype, unit_variance)


class Codebook:
    """Codebook for scalar quantization."""

    def __init__(
        self,
        dim: int,
        bits: int,
        device="cpu",
        dtype=torch.float32,
        unit_variance: bool = True,
        use_beta: bool = False,
    ):
        self.dim = dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        self.num_centroids = 2**bits
        self.centroids = get_centroids(
            dim, bits, device, dtype, unit_variance, use_beta
        )

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize values to nearest centroid indices.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Indices tensor of shape (..., dim) with dtype torch.int64
        """
        # Find nearest centroid for each value
        # x: (..., dim), centroids: (num_centroids,)
        x_expanded = x.unsqueeze(-1)  # (..., dim, 1)
        centroids_expanded = self.centroids.view(1, 1, -1)  # (1, 1, num_centroids)

        distances = (x_expanded - centroids_expanded) ** 2  # (..., dim, num_centroids)
        indices = torch.argmin(distances, dim=-1)  # (..., dim)

        return indices.to(torch.int64)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices to centroid values.

        Args:
            indices: Indices tensor of shape (..., dim)

        Returns:
            Values tensor of shape (..., dim)
        """
        return self.centroids[indices]
