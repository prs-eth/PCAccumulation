"""Minimal binary tensor format shared by the Python tools and the C++ project.

Layout (little endian):
    int32  dtype    0 = float32, 1 = int32
    int32  ndim
    int32  dims[ndim]
    data   prod(dims) elements, C order
"""
import numpy as np

_CODES = {np.dtype(np.float32): 0, np.dtype(np.int32): 1}
_DTYPES = {v: k for k, v in _CODES.items()}


def write_bin(path, arr):
    arr = np.asarray(arr)
    if arr.dtype.kind == 'f':
        arr = arr.astype(np.float32)
    elif arr.dtype.kind in 'iub':
        arr = arr.astype(np.int32)
    arr = np.ascontiguousarray(arr)
    with open(path, 'wb') as f:
        np.array([_CODES[arr.dtype], arr.ndim, *arr.shape], dtype='<i4').tofile(f)
        arr.astype(arr.dtype.newbyteorder('<')).tofile(f)


def read_bin(path):
    with open(path, 'rb') as f:
        code, ndim = np.fromfile(f, dtype='<i4', count=2)
        shape = tuple(np.fromfile(f, dtype='<i4', count=ndim))
        dtype = _DTYPES[int(code)].newbyteorder('<')
        data = np.fromfile(f, dtype=dtype, count=int(np.prod(shape)) if shape else 1)
    return data.reshape(shape).astype(dtype.newbyteorder('='))
