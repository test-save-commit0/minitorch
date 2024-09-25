from typing import Callable, Optional
import numba
from numba import cuda
from .tensor import Tensor
from .tensor_data import MAX_DIMS, Shape, Storage, Strides, TensorData, broadcast_index, index_to_position, shape_broadcast, to_index
from .tensor_ops import MapProto, TensorOps
to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)
THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) ->MapProto:
        """See `tensor_ops.py`"""
        def _map(out: Storage, out_shape: Shape, out_strides: Strides,
                 in_storage: Storage, in_shape: Shape, in_strides: Strides) ->None:
            return tensor_map(fn)(out, out_shape, out_strides,
                                  in_storage, in_shape, in_strides)
        return _map


def tensor_map(fn: Callable[[float], float]) ->Callable[[Storage, Shape,
    Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """
    def _map(out: Storage, out_shape: Shape, out_strides: Strides,
             in_storage: Storage, in_shape: Shape, in_strides: Strides) ->None:
        i = cuda.grid(1)
        if i < len(out):
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(in_storage[in_pos])
    return cuda.jit()(_map)


def tensor_zip(fn: Callable[[float, float], float]) ->Callable[[Storage,
    Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """
    def _zip(out: Storage, out_shape: Shape, out_strides: Strides,
             a_storage: Storage, a_shape: Shape, a_strides: Strides,
             b_storage: Storage, b_shape: Shape, b_strides: Strides) ->None:
        i = cuda.grid(1)
        if i < len(out):
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
    return cuda.jit()(_zip)


def _sum_practice(out: Storage, a: Storage, size: int) ->None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // 	ext{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    shared = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
    i = cuda.grid(1)
    local_i = cuda.threadIdx.x
    shared[local_i] = a[i] if i < size else 0.0
    cuda.syncthreads()

    for s in [1, 2, 4, 8, 16]:
        if local_i % (2 * s) == 0 and i + s < size:
            shared[local_i] += shared[local_i + s]
        cuda.syncthreads()

    if local_i == 0:
        out[cuda.blockIdx.x] = shared[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def tensor_reduce(fn: Callable[[float, float], float]) ->Callable[[Storage,
    Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """
    def _reduce(out: Storage, out_shape: Shape, out_strides: Strides,
                a_storage: Storage, a_shape: Shape, a_strides: Strides,
                reduce_dim: int) ->None:
        shared = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
        i = cuda.grid(1)
        local_i = cuda.threadIdx.x

        if i < len(out):
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            for j in range(MAX_DIMS):
                a_index[j] = out_index[j]

            reduce_size = a_shape[reduce_dim]
            reduce_stride = a_strides[reduce_dim]

            acc = a_storage[index_to_position(a_index, a_strides)]
            for j in range(1, reduce_size):
                a_index[reduce_dim] = j
                acc = fn(acc, a_storage[index_to_position(a_index, a_strides)])

            shared[local_i] = acc
            cuda.syncthreads()

            for s in [1, 2, 4, 8, 16]:
                if local_i % (2 * s) == 0 and local_i + s < THREADS_PER_BLOCK:
                    shared[local_i] = fn(shared[local_i], shared[local_i + s])
                cuda.syncthreads()

            if local_i == 0:
                out[out_pos] = shared[0]

    return cuda.jit()(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) ->None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    shared_a = cuda.shared.array((32, 32), numba.float64)
    shared_b = cuda.shared.array((32, 32), numba.float64)

    i, j = cuda.grid(2)
    if i < size and j < size:
        shared_a[i, j] = a[i * size + j]
        shared_b[i, j] = b[i * size + j]
    cuda.syncthreads()

    if i < size and j < size:
        acc = 0.0
        for k in range(size):
            acc += shared_a[i, k] * shared_b[k, j]
        out[i * size + j] = acc


jit_mm_practice = cuda.jit()(_mm_practice)


def _tensor_matrix_multiply(out: Storage, out_shape: Shape, out_strides:
    Strides, out_size: int, a_storage: Storage, a_shape: Shape, a_strides:
    Strides, b_storage: Storage, b_shape: Shape, b_strides: Strides) ->None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if len(a_shape) > 2 else 0
    b_batch_stride = b_strides[0] if len(b_shape) > 2 else 0

    BLOCK_DIM = 32
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    batch_id, i, j = cuda.grid(3)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    if batch_id < out_shape[0] and i < out_shape[1] and j < out_shape[2]:
        acc = 0.0
        for k_start in range(0, a_shape[-1], BLOCK_DIM):
            if tx < min(BLOCK_DIM, a_shape[-1] - k_start) and i < a_shape[-2]:
                shared_a[ty, tx] = a_storage[batch_id * a_batch_stride + i * a_strides[-2] + (k_start + tx) * a_strides[-1]]
            else:
                shared_a[ty, tx] = 0.0

            if ty < min(BLOCK_DIM, b_shape[-1] - j) and k_start + tx < b_shape[-2]:
                shared_b[tx, ty] = b_storage[batch_id * b_batch_stride + (k_start + tx) * b_strides[-2] + j * b_strides[-1]]
            else:
                shared_b[tx, ty] = 0.0

            cuda.syncthreads()

            for k in range(min(BLOCK_DIM, a_shape[-1] - k_start)):
                acc += shared_a[ty, k] * shared_b[k, tx]

            cuda.syncthreads()

        out[batch_id * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
