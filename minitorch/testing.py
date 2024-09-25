from typing import Callable, Generic, Iterable, Tuple, TypeVar
import minitorch.operators as operators
A = TypeVar('A')


class MathTest(Generic[A]):

    @staticmethod
    def neg(a: A) ->A:
        """Negate the argument"""
        return -a

    @staticmethod
    def addConstant(a: A) ->A:
        """Add contant to the argument"""
        return a + 5

    @staticmethod
    def square(a: A) ->A:
        """Manual square"""
        return a * a

    @staticmethod
    def cube(a: A) ->A:
        """Manual cube"""
        return a * a * a

    @staticmethod
    def subConstant(a: A) ->A:
        """Subtract a constant from the argument"""
        return a - 5

    @staticmethod
    def multConstant(a: A) ->A:
        """Multiply a constant to the argument"""
        return a * 5

    @staticmethod
    def div(a: A) ->A:
        """Divide by a constant"""
        return a / 5

    @staticmethod
    def inv(a: A) ->A:
        """Invert after adding"""
        return 1 / (a + 10)

    @staticmethod
    def sig(a: A) ->A:
        """Apply sigmoid"""
        return 1 / (1 + operators.exp(-a))

    @staticmethod
    def log(a: A) ->A:
        """Apply log to a large value"""
        return operators.log(a + 100)

    @staticmethod
    def relu(a: A) ->A:
        """Apply relu"""
        return max(a, 0)

    @staticmethod
    def exp(a: A) ->A:
        """Apply exp to a smaller value"""
        return operators.exp(a / 100)

    @staticmethod
    def add2(a: A, b: A) ->A:
        """Add two arguments"""
        return a + b

    @staticmethod
    def mul2(a: A, b: A) ->A:
        """Mul two arguments"""
        return a * b

    @staticmethod
    def div2(a: A, b: A) ->A:
        """Divide two arguments"""
        return a / b

    @classmethod
    def _tests(cls) ->Tuple[Tuple[str, Callable[[A], A]], Tuple[str,
        Callable[[A, A], A]], Tuple[str, Callable[[Iterable[A]], A]]]:
        """
        Returns a list of all the math tests.
        """
        one_arg = [
            ("neg", cls.neg),
            ("addConstant", cls.addConstant),
            ("square", cls.square),
            ("cube", cls.cube),
            ("subConstant", cls.subConstant),
            ("multConstant", cls.multConstant),
            ("div", cls.div),
            ("inv", cls.inv),
            ("sig", cls.sig),
            ("log", cls.log),
            ("relu", cls.relu),
            ("exp", cls.exp)
        ]
        two_arg = [
            ("add2", cls.add2),
            ("mul2", cls.mul2),
            ("div2", cls.div2)
        ]
        return (one_arg, two_arg, [])


class MathTestVariable(MathTest):
    pass
