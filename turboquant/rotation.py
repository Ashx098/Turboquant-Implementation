"""Random orthogonal rotation matrix using QR decomposition."""

import torch
import torch.nn.functional as F


def generate_orthogonal_matrix(
    dim: int, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    compute_dtype = torch.float32
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    A = torch.randn(dim, dim, device=device, dtype=compute_dtype, generator=generator)

    # QR decomposition
    Q, R = torch.linalg.qr(A)

    # Adjust signs to ensure positive diagonal
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1

    Pi = Q * signs.unsqueeze(0)

    # Cast to target dtype
    return Pi.to(dtype)


class Rotator:
    """Rotation handler with caching."""

    def __init__(self, dim: int, device="cpu", dtype=torch.float32):
        """Initialize rotator.

        Args:
            dim: Vector dimension
            device: Device for computations
            dtype: Data type
        """
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.Pi = generate_orthogonal_matrix(dim, device, dtype)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        Pi = self.Pi.to(device=x.device, dtype=x.dtype)
        x_rotated = x_flat @ Pi.T
        return x_rotated.reshape(original_shape)

    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        Pi = self.Pi.to(device=x.device, dtype=x.dtype)
        x_unrotated = x_flat @ Pi
        return x_unrotated.reshape(original_shape)
