"""Bit packing for quantized indices.

Supports packing for 1, 2, 3, 4, 8 bit widths.
"""

import torch


def pack_1bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 1-bit indices into uint8 bytes (8 values per byte)."""
    original_shape = indices.shape
    last_dim = original_shape[-1]
    
    indices_flat = indices.reshape(-1, last_dim)
    batch_size = indices_flat.shape[0]
    
    # Pad to multiple of 8
    pad = (8 - last_dim % 8) % 8
    if pad > 0:
        padding = torch.zeros(batch_size, pad, dtype=indices.dtype, device=indices.device)
        indices_flat = torch.cat([indices_flat, padding], dim=1)
        padded_len = last_dim + pad
    else:
        padded_len = last_dim
    
    # Pack 8 values into each byte
    indices_groups = indices_flat.reshape(batch_size, padded_len // 8, 8)
    
    packed = torch.zeros(batch_size, padded_len // 8, dtype=torch.uint8, device=indices.device)
    for i in range(8):
        packed |= (indices_groups[:, :, i].to(torch.uint8) & 0x1) << (7 - i)
    
    new_shape = list(original_shape[:-1]) + [padded_len // 8]
    return packed.reshape(new_shape)


def unpack_1bit(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """Unpack 1-bit indices from uint8 bytes."""
    target_last_dim = original_shape[-1]
    
    packed_flat = packed.reshape(-1, packed.shape[-1])
    batch_size = packed_flat.shape[0]
    
    packed_uint8 = packed_flat.to(torch.uint8)
    
    unpacked = torch.zeros(batch_size, packed_flat.shape[1] * 8, dtype=torch.int64, device=packed.device)
    for i in range(8):
        unpacked[:, i::8] = ((packed_uint8 >> (7 - i)) & 0x1).to(torch.int64)
    
    if target_last_dim % 8 != 0:
        unpacked = unpacked[:, :target_last_dim]
    
    return unpacked.reshape(original_shape)


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices into uint8 bytes (4 values per byte)."""
    original_shape = indices.shape
    last_dim = original_shape[-1]
    
    indices_flat = indices.reshape(-1, last_dim)
    batch_size = indices_flat.shape[0]
    
    # Pad to multiple of 4
    pad = (4 - last_dim % 4) % 4
    if pad > 0:
        padding = torch.zeros(batch_size, pad, dtype=indices.dtype, device=indices.device)
        indices_flat = torch.cat([indices_flat, padding], dim=1)
        padded_len = last_dim + pad
    else:
        padded_len = last_dim
    
    # Pack 4 values into each byte
    indices_groups = indices_flat.reshape(batch_size, padded_len // 4, 4)
    
    packed = torch.zeros(batch_size, padded_len // 4, dtype=torch.uint8, device=indices.device)
    for i in range(4):
        packed |= (indices_groups[:, :, i].to(torch.uint8) & 0x3) << (6 - i * 2)
    
    new_shape = list(original_shape[:-1]) + [padded_len // 4]
    return packed.reshape(new_shape)


def unpack_2bit(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """Unpack 2-bit indices from uint8 bytes."""
    target_last_dim = original_shape[-1]
    
    packed_flat = packed.reshape(-1, packed.shape[-1])
    batch_size = packed_flat.shape[0]
    
    packed_uint8 = packed_flat.to(torch.uint8)
    
    unpacked = torch.zeros(batch_size, packed_flat.shape[1] * 4, dtype=torch.int64, device=packed.device)
    for i in range(4):
        unpacked[:, i::4] = ((packed_uint8 >> (6 - i * 2)) & 0x3).to(torch.int64)
    
    if target_last_dim % 4 != 0:
        unpacked = unpacked[:, :target_last_dim]
    
    return unpacked.reshape(original_shape)


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices into uint8 bytes (2 values per 6 bytes, complex packing)."""
    original_shape = indices.shape
    last_dim = original_shape[-1]
    
    indices_flat = indices.reshape(-1, last_dim)
    batch_size = indices_flat.shape[0]
    
    # For simplicity, pack 2 3-bit values into 1 byte (wasting 2 bits)
    # Real implementation would use tighter packing
    pad = (2 - last_dim % 2) % 2
    if pad > 0:
        padding = torch.zeros(batch_size, pad, dtype=indices.dtype, device=indices.device)
        indices_flat = torch.cat([indices_flat, padding], dim=1)
        padded_len = last_dim + pad
    else:
        padded_len = last_dim
    
    indices_pairs = indices_flat.reshape(batch_size, padded_len // 2, 2)
    
    # Pack: high 4 bits = first (wasting 1 bit), low 4 bits = second
    packed = ((indices_pairs[:, :, 0] & 0x7) << 4) | (indices_pairs[:, :, 1] & 0x7)
    packed = packed.to(torch.uint8)
    
    new_shape = list(original_shape[:-1]) + [padded_len // 2]
    return packed.reshape(new_shape)


def unpack_3bit(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """Unpack 3-bit indices from uint8 bytes."""
    target_last_dim = original_shape[-1]
    
    packed_flat = packed.reshape(-1, packed.shape[-1])
    batch_size = packed_flat.shape[0]
    
    packed_int = packed_flat.to(torch.int32)
    
    high = (packed_int >> 4) & 0x7
    low = packed_int & 0x7
    
    unpacked_pairs = torch.stack([high, low], dim=2)
    unpacked_flat = unpacked_pairs.reshape(batch_size, -1)
    
    if target_last_dim % 2 == 1:
        unpacked_flat = unpacked_flat[:, :target_last_dim]
    
    return unpacked_flat.reshape(original_shape)


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices into uint8 bytes (2 values per byte)."""
    original_shape = indices.shape
    last_dim = original_shape[-1]
    
    indices_flat = indices.reshape(-1, last_dim)
    batch_size = indices_flat.shape[0]
    
    # Handle odd length by padding
    if last_dim % 2 == 1:
        padding = torch.zeros(batch_size, 1, dtype=indices.dtype, device=indices.device)
        indices_flat = torch.cat([indices_flat, padding], dim=1)
        padded_len = last_dim + 1
    else:
        padded_len = last_dim
    
    # Reshape to pairs: (batch, n/2, 2)
    indices_pairs = indices_flat.reshape(batch_size, padded_len // 2, 2)
    
    # Pack: high nibble = first, low nibble = second
    packed = (indices_pairs[:, :, 0] << 4) | indices_pairs[:, :, 1]
    packed = packed.to(torch.uint8)
    
    new_shape = list(original_shape[:-1]) + [padded_len // 2]
    packed = packed.reshape(new_shape)
    
    return packed


def unpack_4bit(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """Unpack 4-bit indices from uint8 bytes."""
    target_last_dim = original_shape[-1]
    
    packed_flat = packed.reshape(-1, packed.shape[-1])
    batch_size = packed_flat.shape[0]
    
    packed_int = packed_flat.to(torch.int32)
    
    high_nibble = (packed_int >> 4) & 0xF
    low_nibble = packed_int & 0xF
    
    unpacked_pairs = torch.stack([high_nibble, low_nibble], dim=2)
    unpacked_flat = unpacked_pairs.reshape(batch_size, -1)
    
    if target_last_dim % 2 == 1:
        unpacked_flat = unpacked_flat[:, :target_last_dim]
    
    unpacked = unpacked_flat.reshape(original_shape)
    
    return unpacked


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack indices according to bit width."""
    if bits == 1:
        return pack_1bit(indices)
    elif bits == 2:
        return pack_2bit(indices)
    elif bits == 3:
        return pack_3bit(indices)
    elif bits == 4:
        return pack_4bit(indices)
    elif bits == 8:
        return indices.to(torch.uint8)
    elif bits == 16:
        return indices.to(torch.uint16)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def unpack_indices(packed: torch.Tensor, original_shape: tuple, bits: int) -> torch.Tensor:
    """Unpack indices according to bit width."""
    if bits == 1:
        return unpack_1bit(packed, original_shape)
    elif bits == 2:
        return unpack_2bit(packed, original_shape)
    elif bits == 3:
        return unpack_3bit(packed, original_shape)
    elif bits == 4:
        return unpack_4bit(packed, original_shape)
    elif bits == 8:
        return packed.to(torch.int64)
    elif bits == 16:
        return packed.to(torch.int64)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def compute_packed_size(num_elements: int, bits: int) -> int:
    """Compute packed size in bytes."""
    if bits == 1:
        return (num_elements + 7) // 8
    elif bits == 2:
        return (num_elements + 3) // 4
    elif bits == 3:
        return (num_elements + 1) // 2  # 2 values per byte (wasting bits)
    elif bits == 4:
        return (num_elements + 1) // 2
    elif bits == 8:
        return num_elements
    elif bits == 16:
        return num_elements * 2
    else:
        raise ValueError(f"Unsupported bit width: {bits}")
