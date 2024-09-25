import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor

from .tensor_strategies import tensors


@pytest.mark.task4_1
def test_conv1d_simple() -> None:
    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    out = minitorch.Conv1dFun.apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    print(input, weight)
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@settings(max_examples=10)
def test_conv_batch(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@settings(max_examples=10)
def test_conv_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
def test_conv2() -> None:
    t = minitorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t.requires_grad_(True)

    t2 = minitorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)


@pytest.mark.task4_1
def test_conv1d_multi_channel_batch() -> None:
    t = minitorch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]).view(2, 2, 3)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).view(2, 2, 2)
    out = minitorch.Conv1dFun.apply(t, t2)

    assert out.shape == (2, 2, 2)
    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 3 * 3 + 4 * 4
    assert out[1, 1, 1] == 7 * 5 + 8 * 6 + 10 * 7 + 11 * 8
    minitorch.grad_check(minitorch.Conv1dFun.apply, t, t2)


@pytest.mark.task4_2
def test_conv2d_multi_channel_batch() -> None:
    t = minitorch.tensor([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                           [[9, 10, 11], [12, 13, 14], [15, 16, 17]]],
                          [[[18, 19, 20], [21, 22, 23], [24, 25, 26]],
                           [[27, 28, 29], [30, 31, 32], [33, 34, 35]]]]).view(2, 2, 3, 3)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                           [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]).view(2, 2, 2, 2)
    out = minitorch.Conv2dFun.apply(t, t2)

    assert out.shape == (2, 2, 2, 2)
    assert out[0, 0, 0, 0] == (0*1 + 1*2 + 3*3 + 4*4) + (9*5 + 10*6 + 12*7 + 13*8)
    assert out[1, 1, 1, 1] == (25*9 + 26*10 + 34*11 + 35*12) + (34*13 + 35*14 + 43*15 + 44*16)
    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)
