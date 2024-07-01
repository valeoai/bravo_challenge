# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

"""
Bravo CODEC: Compression scheme for segmentation maps with confidence values

This module provides functionalities for compressing and decompressing 2D arrays
representing segmentation maps with confidence values. The input arrays are
assumed to be 2D arrays. The scheme is intended to be used on the BRAVO Challenge.

The main functions in this module are `bravo_encode` and `bravo_decode`.

Functions
---------
- `bravo_encode(class_array, confidence_array, ...)`
    Encode a 2D array of class labels and a 2D array of confidence values into
    a compressed byte-string.

- `bravo_decode(encoded_bytes)`
    Decode a BRAVO compressed byte-string back into a 2D array of class labels
    and a 2D array of confidence values.

Usage
-----
    from bravo_codec import bravo_encode, bravo_decode

    class_array, confidence_array = your_segmentation_method(input_image)

    # Encoding
    encoded_bytes = bravo_encode(class_array, confidence_array)

    # Decoding
    decoded_class_array, decoded_confidence_array, header = bravo_decode(encoded_bytes)

Notes
-----
- The class_array compression is lossless.
- The confidence_array compression is lossy, but the loss is controlled by the
  quantization parameters. Use the default values for the BRAVO Challenge.
"""
import struct
from typing import Tuple
import zlib

import numpy as np
import numpy.typing as npt
import zstandard as zstd


HEADER_MAGIC = b"BV23"
HEADER_VERSION = 2
HEADER_FORMAT = "<4sIIIIII"
COMPRESS_TECHNIQUE = 2
COMPRESS_LEVEL = 9  # Compression level for zlib and zstd, from 1 to 9, -1 for default


def _compress(data: bytes) -> bytes:
    if COMPRESS_TECHNIQUE == 1:
        data = zlib.compress(data, level=COMPRESS_LEVEL)
    elif COMPRESS_TECHNIQUE == 2:
        cctx = zstd.ZstdCompressor(level=COMPRESS_LEVEL)
        data = cctx.compress(data)
    else:
        assert False, "Invalid compression technique"
    return data


def _decompress(data: bytes) -> bytes:
    if COMPRESS_TECHNIQUE == 1:
        data = zlib.decompress(data)
    elif COMPRESS_TECHNIQUE == 2:
        dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(data)
    else:
        assert False, "Invalid compression technique"
    return data


def bravo_encode(class_array: npt.NDArray[np.uint8],
                 confidence_array: npt.NDArray[np.uint16],
                 confidence_indices: npt.NDArray[np.uint32]) -> bytes:
    """
    Encode a class array and confidence array into a BRAVO compressed byte-string.

    Parameters
    ----------
    class_array : np.ndarray[np.uint8]
        Array with class labels. Must be 2D.
    confidence_array : np.ndarray[np.uint16]
    confidence_indices : np.ndarray[np.uint32]
        Array with the indices of the sampled confidence values. Must be 1D. Assumed to be sorted.

    Returns
    -------
    bytes
        Compressed byte-string
    """
    # Checks input
    if class_array.ndim != 2:
        raise ValueError("class_array must be 2D")
    if class_array.dtype != np.uint8:
        raise ValueError("class_array must be of dtype np.uint8")
    if confidence_array.ndim != 2:
        raise ValueError("confidence_array must be 2D")
    if confidence_array.dtype != np.uint16:
        raise ValueError("confidence_array must be of dtype np.uint16")
    if confidence_indices is not None:
        if confidence_indices.ndim != 1:
            raise ValueError("confidence_indices must be 1D")
        if confidence_indices.dtype not in (np.uint32, np.int32, np.uint64, np.int64):
            raise ValueError("confidence_indices must be of a large integral type")
    if class_array.shape != confidence_array.shape:
        raise ValueError("class_array and confidence_array must have the same shape")

    # Downsamples the confidence array if necessary
    class_rows = class_array.shape[0]
    class_cols = class_array.shape[1]

    # Gets class array bytes
    class_array = class_array.ravel()
    class_bytes = class_array.tobytes()

    # Gets confidence_array bytes
    confidence_array = confidence_array.ravel()
    if confidence_indices is not None:
        confidence_array = confidence_array[confidence_indices]
    confidence_bytes = confidence_array.tobytes()

    # Compresses both arrays
    data = class_bytes + confidence_bytes
    data = _compress(data)

    # Assembles the header with struct
    header = struct.pack(
            HEADER_FORMAT,
            HEADER_MAGIC,
            HEADER_VERSION,
            class_rows,
            class_cols,
            confidence_indices.size if confidence_indices is not None else 0,
            len(class_bytes),
            len(confidence_bytes)
        )

    data = header + data
    crc32 = zlib.crc32(data)

    # Returns the compressed byte-string
    return data + struct.pack("<I", crc32)


def bravo_decode(encoded_bytes: bytes,
                 dequantize: bool = False) -> Tuple[npt.NDArray[np.uint8], np.ndarray, dict]:
    """
    Decode a BRAVO compressed byte-string into a class array and confidence array. The confidence array is NOT upsampled
    to the original size, and downstream processing should take care of this if needed.

    Parameters
    ----------
    encoded_bytes : bytes
        The compressed byte-string.
    dequantize : bool, default = False
        If True, the confidence array is dequantized to np.float32. If False, the confidence array is kept as np.uint16.

    Returns
    -------
    np.ndarray[np.uint8]
        The class array.
    np.ndarray
        The confidence array.
        If dequantize=True and quantize_levels > 0, the confidence array is restored to its original values as
        np.float32.
        If quantize_levels == 0, the confidence array is kept as the original np.uint8.
    dict(str, Any)
        The header information.
    """

    # Parse the header
    header_size = struct.calcsize(HEADER_FORMAT)
    header_bytes = encoded_bytes[:header_size]
    header = struct.unpack(HEADER_FORMAT, header_bytes)
    signature, version, class_rows, class_cols, confidence_size, class_len,  confidence_len = header

    # Check the signature and version
    if signature != HEADER_MAGIC:
        raise ValueError("Invalid magic number in header")
    if version != HEADER_VERSION:
        raise ValueError("Invalid version number in header")

    # Check the CRC32
    crc32 = struct.unpack("<I", encoded_bytes[-4:])[0]
    crc32_check = zlib.crc32(encoded_bytes[:-4])
    if crc32 != crc32_check:
        raise ValueError("CRC32 check failed")

    # Decompress the class and confidence arrays
    data = _decompress(encoded_bytes[header_size:-4])
    if len(data) != class_len + confidence_len:
        raise ValueError("Invalid lengths in header")
    class_bytes = data[:class_len]
    confidence_bytes = data[class_len:]

    # Reconstruct class array
    class_array = np.frombuffer(class_bytes, dtype=np.uint8).reshape((class_rows, class_cols))

    # Reconstruct confidence array
    confidence_array: np.ndarray = np.frombuffer(confidence_bytes, dtype=np.uint16)

    # Dequantize the confidence array
    if dequantize:
        confidence_array = (confidence_array.astype(np.float32) + 0.5) / 65536.0

    return_header = {
        "class_rows": class_rows,
        "class_cols": class_cols,
        "confidence_size": confidence_size,
        "quantize_levels": 65536,
        "quantize_classes": 0,
    }
    return class_array, confidence_array, return_header


def get_quantization_levels(quantize_levels: int = 65536,
                            quantize_classes: int = 0,  # pylint: disable=unused-argument
                            **_kwargs) -> np.ndarray:
    '''
    Get the float values of the quantization levels for the confidence array.

    Args:
    - quantize_levels: int, default = 128
        Number of quantization levels.
    - quantize_classes: int, default = 19
        Number of classes, ignored, included for backwards compatibility.
    - **_kwargs: dict
        Ignored keyword arguments, included for convenience for allowing calling this function with the **header
        dictionary returned by bravo_decode.

    Returns:
    - np.ndarray
        The quantization levels as float values.
    '''
    levels = (np.linspace(0., quantize_levels-1, quantize_levels, dtype=np.float32) + 0.5) / quantize_levels
    return levels
