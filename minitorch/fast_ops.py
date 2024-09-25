from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit, prange
from .tensor_data import MAX_DIMS, broadcast_index, index_to_position, shape_broadcast, to_index
from .tensor_ops import MapProto, TensorOps
if TYPE_CHECKING:
    from typing import Callable, Optional
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides
to_index = njit(inline='always')(to_index)
index_to_position = njit(inline='always')(index_to_position)
broadcast_index = njit(inline='always')(broadcast_index)


class FastOps(TensorOps):

    @staticmethod
    def map(fn: Callable[[float], float]) ->MapProto:
        """See `tensor_ops.py`"""
        def _map(out: Tensor, out_shape: Shape, out_strides: Strides, in_storage: Storage, in_shape: Shape, in_strides: Strides) -> None:
            return tensor_map(fn)(out._tensor._storage, out_shape, out_strides, in_storage, in_shape, in_strides)
        return _map

    @staticmethod
    def zip(fn: Callable[[float, float], float]) ->Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        def _zip(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            tensor_zip(fn)(
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                b._tensor._storage,
                b._tensor._shape,
                b._tensor._strides,
            )
            return out
        return _zip

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float=0.0) ->Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        def _reduce(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            tensor_reduce(fn)(
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                dim,
            )
            return out
        return _reduce

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) ->Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """
        # Make sure the last dimension of a matches the second-to-last dimension of b
        assert a.shape[-1] == b.shape[-2]

        # Calculate the output shape
        out_shape = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        out_shape.append(a.shape[-2])
        out_shape.append(b.shape[-1])

        # Create the output tensor
        out = a.zeros(tuple(out_shape))

        # Perform matrix multiplication
        tensor_matrix_multiply(
            out._tensor._storage,
            out._tensor._shape,
            out._tensor._strides,
            a._tensor._storage,
            a._tensor._shape,
            a._tensor._strides,
            b._tensor._storage,
            b._tensor._shape,
            b._tensor._strides,
        )

        return out


def tensor_map(fn: Callable[[float], float]) ->Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """
    @njit(parallel=True)
    def _map(out: Storage, out_shape: Shape, out_strides: Strides,
             in_storage: Storage, in_shape: Shape, in_strides: Strides) -> None:
        # Calculate total number of elements
        size = int(np.prod(in_shape))

        # Check if the tensors are stride-aligned
        is_stride_aligned = np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape)

        if is_stride_aligned:
            # If stride-aligned, we can avoid indexing
            for i in prange(size):
                out[i] = fn(in_storage[i])
        else:
            # If not stride-aligned, we need to use indexing
            for i in prange(size):
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                in_index = np.zeros(len(in_shape), dtype=np.int32)
                to_index(i, in_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])

    return _map


def tensor_zip(fn: Callable[[float, float], float]) ->Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """
    @njit(parallel=True)
    def _zip(out: Storage, out_shape: Shape, out_strides: Strides,
             a_storage: Storage, a_shape: Shape, a_strides: Strides,
             b_storage: Storage, b_shape: Shape, b_strides: Strides) -> None:
        # Calculate total number of elements
        size = int(np.prod(out_shape))

        # Check if all tensors are stride-aligned
        is_stride_aligned = (np.array_equal(out_strides, a_strides) and
                             np.array_equal(out_strides, b_strides) and
                             np.array_equal(out_shape, a_shape) and
                             np.array_equal(out_shape, b_shape))

        if is_stride_aligned:
            # If stride-aligned, we can avoid indexing
            for i in prange(size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # If not stride-aligned, we need to use indexing
            for i in prange(size):
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                a_index = np.zeros(len(a_shape), dtype=np.int32)
                b_index = np.zeros(len(b_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) ->Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """
    @njit(parallel=True)
    def _reduce(out: Storage, out_shape: Shape, out_strides: Strides,
                a_storage: Storage, a_shape: Shape, a_strides: Strides,
                reduce_dim: int) -> None:
        # Calculate sizes
        reduce_size = a_shape[reduce_dim]
        non_reduce_size = int(np.prod([s for i, s in enumerate(a_shape) if i != reduce_dim]))

        # Main loop in parallel
        for i in prange(non_reduce_size):
            # Calculate the base index for the non-reduce dimensions
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            # Inner reduction loop
            acc = out[out_pos]  # Initialize accumulator with the starting value
            for j in range(reduce_size):
                # Calculate the full index for the input tensor
                a_index = out_index.copy()
                a_index = np.insert(a_index, reduce_dim, j)
                a_pos = index_to_position(a_index, a_strides)
                acc = fn(acc, a_storage[a_pos])

            # Store the result
            out[out_pos] = acc

    return _reduce


def _tensor_matrix_multiply(out: Storage, out_shape: Shape, out_strides: Strides,
                            a_storage: Storage, a_shape: Shape, a_strides: Strides,
                            b_storage: Storage, b_shape: Shape, b_strides: Strides) ->None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if len(a_shape) == 3 else 0
    b_batch_stride = b_strides[0] if len(b_shape) == 3 else 0
    
    # Outer loop in parallel
    for i in prange(out_shape[0]):
        for j in range(out_shape[1]):
            for k in range(out_shape[2]):
                a_inner = 0.0
                for l in range(a_shape[-1]):
                    a_index = i * a_batch_stride + j * a_strides[-2] + l * a_strides[-1]
                    b_index = i * b_batch_stride + l * b_strides[-2] + k * b_strides[-1]
                    a_inner += a_storage[a_index] * b_storage[b_index]
                out_index = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]
                out[out_index] = a_inner


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(
    _tensor_matrix_multiply)
