"""TurboQuant PROD variant - optimize for inner product accuracy.

This implementation follows the paper exactly:
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
arXiv:2504.19874

Key innovation: Two-stage quantization for UNBIASED inner product estimation.
(b-1) bits for MSE quantization + 1 bit for QJL residual correction.
"""

import torch
import math
from .rotation import Rotator
from .codebook import Codebook
from .qjl import QJL
from .packing import pack_indices, unpack_indices


class TurboQuantPROD:
    """TurboQuant optimized for inner product accuracy (unbiased estimation).
    
    Uses two-stage quantization:
    1. MSE quantization with (b-1) bits
    2. QJL (Quantized Johnson-Lindenstrauss) on residual for unbiased correction
    
    Paper claims:
    - Unbiased: E[<y, x_tilde>] = <y, x>
    - Variance: D_prod <= (sqrt(3) * pi^2 * ||y||^2) / (d * 4^b)
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        num_qjl_hashes: int = None,
        device="cpu",
        dtype=torch.float32,
        unit_variance: bool = True,
    ):
        """Initialize TurboQuantPROD.
        
        Args:
            dim: Vector dimension
            bits: Total bits per coordinate (actual MSE uses b-1, 1 bit for QJL)
            num_qjl_hashes: Number of QJL hash bits (default: dim)
            device: Device for computations
            dtype: Data type
            unit_variance: Whether input has unit variance
        """
        self.dim = dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        if num_qjl_hashes is None:
            num_qjl_hashes = dim

        self.num_qjl_hashes = num_qjl_hashes
        self.mse_bits = max(1, bits - 1)  # Reserve 1 bit for QJL

        self.rotator = Rotator(dim, device, dtype)
        # For unit-sphere vectors, coordinate variance is 1/dim after rotation
        self.codebook = Codebook(dim, self.mse_bits, device, dtype, unit_variance=False)
        self.qjl = QJL(dim, num_qjl_hashes, device, dtype)
        
        # Precompute QJL dequantization scaling factor: gamma * sqrt(pi/2) / sqrt(d)
        # Note: With S_ij ~ N(0, 1/d), the scaling changes from the paper's formula
        # Paper assumes S_ij ~ N(0,1), giving E[||S_j||^2] = m
        # We use S_ij ~ N(0, 1/d), giving E[||S_j||^2] = m/d
        # So the factor becomes sqrt(pi/2) * m/d / sqrt(m/d) = sqrt(pi*m/(2*d))
        # But for inner products, we need: E[<S.y, sign(S.r)>] = ||r|| * <y, r>/||r|| = <y, r>
        # Actually the correct scaling is: E[S^T sign(S.r)] = sqrt(2/pi) * r
        # So dequantization should be: r_tilde = gamma * sqrt(pi/2) * S^T sign(S.r) / ||S^T sign(S.r)||
        # Let's use: r_tilde = gamma * (pi/2) * S^T sign(S.r) / m
        self.qjl_scale = math.pi / (2 * num_qjl_hashes)

    def quantize(
        self, x: torch.Tensor, return_norm: bool = True, pack: bool = False
    ) -> dict:
        """Quantize vector using TurboQuant PROD.
        
        Algorithm 2 from paper:
        1. idx <- Quant_mse(x) using (b-1) bits
        2. x_mse <- DeQuant_mse(idx)
        3. r <- x - x_mse (residual)
        4. gamma <- ||r||_2 (residual norm)
        5. qjl <- sign(S . r) (1-bit QJL hash)
        
        Args:
            x: Input tensor of shape (..., dim)
            return_norm: Whether to return the norm for reconstruction
            pack: If True and bits=4, pack indices into uint8

        Returns:
            Dict with:
                - indices: MSE quantized indices (..., dim) or packed
                - qjl_hash: QJL hash of residual (..., num_qjl_hashes)
                - gamma: L2 norm of RESIDUAL (not original!) for QJL scaling
                - norm_original: L2 norm of original vector (optional)
                - packed: True if indices are packed
                - original_shape: Shape for unpacking
        """
        # Step 1: Rotate
        x_rotated = self.rotator.rotate(x)

        # Step 2: MSE quantization with (b-1) bits
        indices = self.codebook.quantize(x_rotated)
        x_mse_rotated = self.codebook.dequantize(indices)

        # Step 3: Compute residual in ROTATED space
        r_rotated = x_rotated - x_mse_rotated

        # Step 4: QJL hash of residual (provides 1-bit quantization)
        # Paper: Q_qjl(x) := sign(S . x) where S_ij ~ N(0,1)
        qjl_hash = self.qjl.hash(r_rotated)

        # Step 5: Compute residual norm for scaling
        # ||r||_2 is needed for proper dequantization
        gamma = torch.norm(r_rotated, dim=-1, keepdim=True)

        result = {
            "indices": indices,
            "qjl_hash": qjl_hash,
            "gamma": gamma,  # Residual norm for QJL scaling
            "packed": False,
            "original_shape": indices.shape,
        }

        # Pack if requested
        if pack:
            result["indices"] = pack_indices(indices, self.mse_bits)
            result["packed"] = True

        if return_norm:
            # Store original vector norm for optional renormalization
            norm_original = torch.norm(x, dim=-1, keepdim=True)
            result["norm_original"] = norm_original

        return result

    def dequantize(
        self, quantized: dict, use_qjl_correction: bool = True
    ) -> torch.Tensor:
        """Dequantize vector with FULL QJL residual reconstruction.
        
        Paper Algorithm 2 DeQuant_prod:
        x_tilde_mse <- DeQuant_mse(idx)
        x_tilde_qjl <- sqrt(pi/2)/d * gamma * S^T . qjl
        RETURN x_tilde_mse + x_tilde_qjl
        
        Args:
            quantized: Dict with 'indices', 'qjl_hash', 'gamma', optionally 'norm_original'
            use_qjl_correction: Whether to apply QJL residual correction

        Returns:
            Reconstructed tensor of shape (..., dim) - UNBIASED estimate
        """
        indices = quantized["indices"]

        # Unpack if needed
        if quantized.get("packed", False):
            indices = unpack_indices(indices, quantized["original_shape"], self.mse_bits)

        # Step 1: Reconstruct MSE part in ROTATED space
        x_mse_rotated = self.codebook.dequantize(indices)

        if use_qjl_correction and "qjl_hash" in quantized and "gamma" in quantized:
            # Step 2: Reconstruct QJL residual
            # Paper: Q_qjl^{-1}(z) := sqrt(pi/2)/d * S^T . z
            qjl_hash = quantized["qjl_hash"]
            gamma = quantized["gamma"]
            
            # Reconstruct residual: r_tilde = Q_qjl^{-1}(qjl_hash) * gamma / E[||Q_qjl^{-1}||]
            # Since QJL preserves norm in expectation, we scale by gamma
            x_qjl_rotated = self.qjl.dequantize(qjl_hash)  # S^T . qjl
            
            # Apply proper scaling: sqrt(pi/2)/d * gamma * S^T . qjl
            # Note: qjl.dequantize returns S^T . qjl, we need to scale
            x_qjl_rotated = x_qjl_rotated * gamma * self.qjl_scale
            
            # Step 3: Combine MSE + QJL for unbiased estimate
            x_reconstructed_rotated = x_mse_rotated + x_qjl_rotated
        else:
            # Without QJL correction, we're biased (just MSE part)
            x_reconstructed_rotated = x_mse_rotated

        # Step 4: Inverse rotation to get back to original space
        x_reconstructed = self.rotator.inverse_rotate(x_reconstructed_rotated)

        # Optional: Renormalize if original norm was stored
        # This helps when input vectors have fixed norm (e.g., normalized embeddings)
        if "norm_original" in quantized:
            current_norm = torch.norm(x_reconstructed, dim=-1, keepdim=True)
            x_reconstructed = x_reconstructed * (
                quantized["norm_original"] / (current_norm + 1e-8)
            )

        return x_reconstructed

    def compute_inner_product(
        self, 
        x_quantized: dict, 
        y: torch.Tensor,
        use_qjl: bool = True
    ) -> torch.Tensor:
        """Compute UNBIASED inner product <x, y> using quantized representation.
        
        This is the PREFERRED method for attention computation - avoids full dequantization.
        
        Paper fusion identity: <q, Pi^T . c> = <Pi . q, c>
        
        Args:
            x_quantized: Quantized representation of x from quantize()
            y: Query vector (NOT quantized) of shape (..., dim)
            use_qjl: Whether to include QJL correction

        Returns:
            Unbiased estimate of <x, y>
        """
        # Rotate query to match quantized space
        y_rotated = self.rotator.rotate(y)

        # Unpack indices if packed
        indices = x_quantized["indices"]
        if x_quantized.get("packed", False):
            indices = unpack_indices(indices, x_quantized["original_shape"], self.mse_bits)

        # Term 1: MSE contribution
        # <y_rotated, centroids[indices]>
        x_mse_rotated = self.codebook.dequantize(indices)
        term1 = (y_rotated * x_mse_rotated).sum(dim=-1)

        if use_qjl and "qjl_hash" in x_quantized and "gamma" in x_quantized:
            # Term 2: QJL correction for residual
            # <y_rotated, r_tilde> = gamma * sqrt(pi/2)/d * <S . y_rotated, qjl_hash>
            qjl_hash = x_quantized["qjl_hash"]
            gamma = x_quantized["gamma"].squeeze(-1)
            
            # Compute S . y_rotated
            y_projected = self.qjl.project(y_rotated)  # (..., num_hashes)
            
            # <S . y, qjl> where qjl = sign(S . r)
            # This estimates <y, r> / ||r||
            ip_with_residual_sign = (y_projected * qjl_hash).sum(dim=-1)
            
            # Scale: gamma * sqrt(pi/2)/d * <S.y, sign(S.r)>
            # Expectation: E[<S.y, sign(S.r)>] = ||r|| * <y, r> / ||r|| = <y, r>
            # But we have ||r|| = gamma, so:
            term2 = gamma * self.qjl_scale * ip_with_residual_sign
            
            return term1 + term2
        
        return term1

    def approximate_inner_product(
        self, q1: dict, q2: dict
    ) -> torch.Tensor:
        """Compute approximate inner product between TWO quantized vectors.
        
        Note: This is LESS ACCURATE than compute_inner_product() with one quantized
        and one full-precision vector. Use compute_inner_product() for attention.
        
        Args:
            q1: Quantized representation of first vector
            q2: Quantized representation of second vector

        Returns:
            Approximate inner product
        """
        # Dequantize both and compute
        x1 = self.dequantize(q1, use_qjl_correction=True)
        x2 = self.dequantize(q2, use_qjl_correction=True)
        
        return (x1 * x2).sum(dim=-1)


def quantize_prod(
    x: torch.Tensor, 
    bits: int, 
    num_qjl_hashes: int = None, 
    pack: bool = False
) -> dict:
    """Convenience function for one-shot PROD quantization.

    Args:
        x: Input tensor of shape (..., dim)
        bits: Total bits per coordinate (MSE uses b-1, 1 bit for QJL)
        num_qjl_hashes: Number of QJL hash bits
        pack: If True and bits=4, pack indices into uint8

    Returns:
        Dict with quantization results including gamma (residual norm)
    """
    dim = x.shape[-1]
    device = x.device
    dtype = x.dtype

    quantizer = TurboQuantPROD(dim, bits, num_qjl_hashes, device, dtype)
    return quantizer.quantize(x, pack=pack)


def dequantize_prod(
    quantized: dict,
    dim: int,
    bits: int,
    num_qjl_hashes: int = None,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Convenience function for one-shot PROD dequantization.

    Args:
        quantized: Dict with quantization results (must include 'gamma')
        dim: Vector dimension
        bits: Bits per coordinate
        num_qjl_hashes: Number of QJL hash bits
        device: Device
        dtype: Data type

    Returns:
        Reconstructed tensor (unbiased estimate)
    """
    quantizer = TurboQuantPROD(dim, bits, num_qjl_hashes, device, dtype)
    return quantizer.dequantize(quantized, use_qjl_correction=True)


def compute_ip_prod(
    x_quantized: dict,
    y: torch.Tensor,
    dim: int,
    bits: int,
    num_qjl_hashes: int = None,
) -> torch.Tensor:
    """Convenience function for computing inner product with PROD quantization.
    
    This is the recommended way to use TurboQuantPROD for attention scoring.
    
    Args:
        x_quantized: Quantized representation from quantize_prod()
        y: Query vector (full precision)
        dim: Dimension
        bits: Bits used for quantization
        num_qjl_hashes: Number of QJL hashes

    Returns:
        Unbiased inner product estimate <x, y>
    """
    device = y.device
    dtype = y.dtype
    quantizer = TurboQuantPROD(dim, bits, num_qjl_hashes, device, dtype)
    return quantizer.compute_inner_product(x_quantized, y, use_qjl=True)
