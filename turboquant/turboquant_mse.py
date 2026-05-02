"""TurboQuant MSE variant - minimize reconstruction error."""

import torch
from .rotation import Rotator
from .codebook import Codebook
from .packing import pack_indices, unpack_indices


class TurboQuantMSE:
    """TurboQuant with MSE objective."""

    def __init__(
        self,
        dim: int,
        bits: int,
        device="cpu",
        dtype=torch.float32,
        unit_variance: bool = True,
    ):
        self.dim = dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        self.rotator = Rotator(dim, device, dtype)
        self.codebook = Codebook(dim, bits, device, dtype, unit_variance)

    def quantize(self, x: torch.Tensor, pack: bool = False) -> dict:
        """Quantize vector.

        Args:
            x: Input tensor of shape (..., dim)
            pack: If True and bits=4, pack indices into uint8

        Returns:
            Dict with:
                - indices: Quantized indices (..., dim) or packed
                - packed: True if indices are packed
                - original_shape: Shape for unpacking
        """
        x_rotated = self.rotator.rotate(x)
        indices = self.codebook.quantize(x_rotated)

        result = {
            "indices": indices,
            "packed": False,
            "original_shape": indices.shape,
        }

        # Pack if requested
        if pack:
            result["indices"] = pack_indices(indices, self.bits)
            result["packed"] = True

        return result

    def dequantize(self, quantized: dict) -> torch.Tensor:
        """Dequantize vector.

        Args:
            quantized: Dict with 'indices' key, optionally 'packed' and 'original_shape'

        Returns:
            Reconstructed tensor of shape (..., dim)
        """
        indices = quantized["indices"]

        # Unpack if needed
        if quantized.get("packed", False):
            indices = unpack_indices(indices, quantized["original_shape"], self.bits)

        x_rotated_reconstructed = self.codebook.dequantize(indices)
        x_reconstructed = self.rotator.inverse_rotate(x_rotated_reconstructed)

        return x_reconstructed


def quantize_mse(x: torch.Tensor, bits: int, pack: bool = False) -> dict:
    """Convenience function for one-shot MSE quantization.

    Args:
        x: Input tensor of shape (..., dim)
        bits: Bits per coordinate
        pack: If True and bits=4, pack indices into uint8

    Returns:
        Dict with quantization results
    """
    dim = x.shape[-1]
    device = x.device
    dtype = x.dtype

    quantizer = TurboQuantMSE(dim, bits, device, dtype)
    return quantizer.quantize(x, pack=pack)


def dequantize_mse(
    quantized: dict, dim: int, bits: int, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    """Convenience function for one-shot MSE dequantization.

    Args:
        quantized: Dict with 'indices' key
        dim: Vector dimension
        bits: Bits per coordinate
        device: Device
        dtype: Data type

    Returns:
        Reconstructed tensor
    """
    quantizer = TurboQuantMSE(dim, bits, device, dtype)
    return quantizer.dequantize(quantized)
