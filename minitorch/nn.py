from typing import Tuple
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) ->Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    new_height = (height - kh) // 1 + 1
    new_width = (width - kw) // 1 + 1
    
    tiled = input.view(batch, channel, height, 1, width, 1)
    tiled = tiled.unfold(3, kh, 1).unfold(5, kw, 1)
    tiled = tiled.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    
    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) ->Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=-1)
    return pooled


max_reduce = FastOps.reduce(operators.max, -1000000000.0)


def argmax(input: Tensor, dim: int) ->Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    max_vals, _ = input.max(dim=dim, keepdim=True)
    return (input == max_vals).float()


class Max(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) ->Tensor:
        """Forward of max should be max reduction"""
        output = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, dim, output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) ->Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        input, dim, output = ctx.saved_values
        arg_max = argmax(input, int(dim.item()))
        return grad_output * arg_max, 0.0


def softmax(input: Tensor, dim: int) ->Tensor:
    """
    Compute the softmax as a tensor.



    $z_i = \\frac{e^{x_i}}{\\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    max_val = input.max(dim=dim, keepdim=True)[0]
    exp_x = (input - max_val).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def logsoftmax(input: Tensor, dim: int) ->Tensor:
    """
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \\log \\sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    max_val = input.max(dim=dim, keepdim=True)[0]
    exp_x = (input - max_val).exp()
    return input - max_val - (exp_x.sum(dim=dim, keepdim=True)).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) ->Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max_reduce(tiled, dim=-1)
    return pooled


def dropout(input: Tensor, rate: float, ignore: bool=False) ->Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore or rate == 0:
        return input
    
    mask = (rand(input.shape) > rate).float()
    return (input * mask) / (1 - rate)
