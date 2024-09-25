"""
Implementation of the autodifferentiation Functions for Tensor.
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING
import numpy as np
import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend
if TYPE_CHECKING:
    from typing import Any, List, Tuple
    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class Function:
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError()


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        return (grad_output.f.neg_map(grad_output),)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        t1 = ctx.saved_values[0]
        return (grad_output.f.inv_map(t1.f.mul_map(t1)).f.mul_map(grad_output).f.neg_map(),)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return (grad_output, grad_output)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values
        return (grad_output.f.mul_zip(grad_output, t2),
                grad_output.f.mul_zip(grad_output, t1))


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        result = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        sigmoid_output = ctx.saved_values[0]
        return (grad_output.f.mul_zip(grad_output, 
                sigmoid_output.f.mul_zip(sigmoid_output, 
                sigmoid_output.f.add_scalar(-1).f.neg_map())),)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        t1 = ctx.saved_values[0]
        return (grad_output.f.mul_zip(grad_output, t1.f.relu_back_zip(t1)),)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        t1 = ctx.saved_values[0]
        return (grad_output.f.mul_zip(grad_output, t1.f.inv_map(t1)),)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        result = t1.f.exp_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        exp_output = ctx.saved_values[0]
        return (grad_output.f.mul_zip(grad_output, exp_output),)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.sum(int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        original_shape, dim = ctx.saved_values
        grad = grad_output
        for _ in range(len(original_shape) - grad.dims):
            grad = grad.unsqueeze(int(dim.item()))
        grad = grad.expand(original_shape)
        return (grad, None)


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        return a.all(int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        return (None, None)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        return (None, None)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        return (None, None)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.is_close_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        return (None, None)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        ctx.save_for_backward(order)
        return a.permute(*[int(i.item()) for i in order])

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        order = ctx.saved_values[0]
        inv_order = [0] * len(order)
        for i, p in enumerate(order):
            inv_order[int(p.item())] = i
        return (grad_output.permute(*inv_order), None)


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        return a.view(*[int(i.item()) for i in shape])

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        original_shape = ctx.saved_values[0]
        return (grad_output.view(original_shape), None)


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        return (grad_output,)


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values
        grad_t1 = grad_output.f.matrix_multiply(grad_output, t2.transpose())
        grad_t2 = t1.transpose().f.matrix_multiply(t1.transpose(), grad_output)
        return (grad_t1, grad_t2)


def zeros(shape: UserShape, backend: TensorBackend=SimpleBackend) ->Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return Tensor.make([0] * int(operators.prod(shape)), shape, backend=backend)


def rand(shape: UserShape, backend: TensorBackend=SimpleBackend,
    requires_grad: bool=False) ->Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(ls: Any, shape: UserShape, backend: TensorBackend=SimpleBackend,
    requires_grad: bool=False) ->Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(ls: Any, backend: TensorBackend=SimpleBackend, requires_grad:
    bool=False) ->Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    if isinstance(ls, Tensor):
        return ls
    if isinstance(ls, (float, int)):
        return _tensor([ls], (1,), backend=backend, requires_grad=requires_grad)
    if isinstance(ls, (list, tuple)):
        tensor = _tensor(np.array(ls).flatten(), shape=np.array(ls).shape, backend=backend)
        tensor.requires_grad_(requires_grad)
        return tensor
    raise TypeError("Cannot create tensor from %s" % (type(ls)))
