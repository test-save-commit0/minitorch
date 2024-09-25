from typing import Tuple
import numpy as np
from numba import njit, prange
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import MAX_DIMS, Index, Shape, Strides, broadcast_index, index_to_position, to_index
from .tensor_functions import Function
to_index = njit(inline='always')(to_index)
index_to_position = njit(inline='always')(index_to_position)
broadcast_index = njit(inline='always')(broadcast_index)


def _tensor_conv1d(out: Tensor, out_shape: Shape, out_strides: Strides,
    out_size: int, input: Tensor, input_shape: Shape, input_strides:
    Strides, weight: Tensor, weight_shape: Shape, weight_strides: Strides,
    reverse: bool) ->None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels, in_channels, k_width = weight_shape

    for i in prange(out_size):
        out_index = np.empty(3, np.int32)
        to_index(i, out_shape, out_index)
        b, oc, x = out_index

        # Initialize accumulator
        acc = 0.0

        for ic in range(in_channels):
            for k in range(k_width):
                if reverse:
                    w = x + k
                else:
                    w = x - k + k_width - 1

                if 0 <= w < width:
                    in_pos = index_to_position((b, ic, w), input_strides)
                    w_pos = index_to_position((oc, ic, k), weight_strides)
                    acc += input[in_pos] * weight[w_pos]

        out_pos = index_to_position((b, oc, x), out_strides)
        out[out_pos] = acc


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) ->Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x w
            weight : out_channel x in_channel x kw

        Returns:
            batch x out_channel x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        
        # Compute output shape
        ow = w - kw + 1
        
        # Create output tensor
        out = input.zeros((batch, out_channels, ow))
        
        # Call tensor_conv1d
        tensor_conv1d(
            out._tensor._storage,
            out.shape,
            out.strides,
            out._tensor._size,
            input._tensor._storage,
            input.shape,
            input.strides,
            weight._tensor._storage,
            weight.shape,
            weight.strides,
            False
        )
        
        return out


conv1d = Conv1dFun.apply


def _tensor_conv2d(out: Tensor, out_shape: Shape, out_strides: Strides,
    out_size: int, input: Tensor, input_shape: Shape, input_strides:
    Strides, weight: Tensor, weight_shape: Shape, weight_strides: Strides,
    reverse: bool) ->None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch, out_channels, out_height, out_width = out_shape
    batch, in_channels, in_height, in_width = input_shape
    out_channels, in_channels, k_height, k_width = weight_shape

    for i in prange(out_size):
        out_index = np.empty(4, np.int32)
        to_index(i, out_shape, out_index)
        b, oc, y, x = out_index

        # Initialize accumulator
        acc = 0.0

        for ic in range(in_channels):
            for kh in range(k_height):
                for kw in range(k_width):
                    if reverse:
                        h, w = y + kh, x + kw
                    else:
                        h, w = y - kh + k_height - 1, x - kw + k_width - 1

                    if 0 <= h < in_height and 0 <= w < in_width:
                        in_pos = index_to_position((b, ic, h, w), input_strides)
                        w_pos = index_to_position((oc, ic, kh, kw), weight_strides)
                        acc += input[in_pos] * weight[w_pos]

        out_pos = index_to_position((b, oc, y, x), out_strides)
        out[out_pos] = acc


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) ->Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape
        
        # Compute output shape
        oh = h - kh + 1
        ow = w - kw + 1
        
        # Create output tensor
        out = input.zeros((batch, out_channels, oh, ow))
        
        # Call tensor_conv2d
        tensor_conv2d(
            out._tensor._storage,
            out.shape,
            out.strides,
            out._tensor._size,
            input._tensor._storage,
            input.shape,
            input.strides,
            weight._tensor._storage,
            weight.shape,
            weight.strides,
            False
        )
        
        return out


conv2d = Conv2dFun.apply
