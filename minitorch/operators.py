"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers x and y."""
    return x * y


def id(x: float) -> float:
    """Return the same number x."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers x and y."""
    return x + y


def neg(x: float) -> float:
    """Negate a number x."""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 if x is less than y else 0.0."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return 1.0 if x is equal to y else 0.0."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of x and y."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return 1.0 if x is close to y else 0.0."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Return the sigmoid of x."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the relu of x."""
    return max(x, 0.0)


def log(x: float) -> float:
    """Return the log of x."""
    return math.log(x)


def exp(x: float) -> float:
    """Return the exp of x."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Return the derivative of log(x) with respect to x."""
    return d / x


def inv(x: float) -> float:
    """Return the inverse of x."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Return the derivative of inv(x) with respect to x."""
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Return the derivative of relu(x) with respect to x."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(f: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Map a function f over a list ls."""
    return [f(x) for x in ls]


def zipWith(
    f: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Zip two lists together using zipWith and f."""
    return [f(x, y) for x, y in zip(ls1, ls2)]


def reduce(f: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    it = iter(ls)
    try:
        result = next(it)  # 获取第一个元素作为初始值
    except StopIteration:
        return 0

    for item in it:
        result = f(result, item)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map and neg."""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists using zipWith and add."""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce and add."""
    return reduce(add, ls)


def prod(ls: Iterable[float]) -> float:
    """Take the product of all elements in a list using reduce and mul."""
    return reduce(mul, ls)
