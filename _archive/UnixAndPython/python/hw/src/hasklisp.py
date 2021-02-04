#!/usr/bin/env python3

import typing as tp

from collections import namedtuple

Nil = namedtuple("Nil", ())
Cons = namedtuple("Cons", ("car", "cdr"))
List = tp.Union[Nil, Cons]


def null(lst: List) -> bool:
    """
    >>> null(Nil())
    True
    >>> null(Cons(0, Nil()))
    False
    """
    return True if not lst else False


def from_seq(seq: tp.Sequence) -> List:
    """
    >>> from_seq([])
    Nil()
    >>> from_seq(tuple())
    Nil()
    >>> from_seq([1, 2, 3])
    Cons(car=1, cdr=Cons(car=2, cdr=Cons(car=3, cdr=Nil())))
    """
    if not seq:
        return Nil()
    else:
        return Cons(car=seq[0], cdr=from_seq(seq[1:]))


# print(from_seq([]))
# print(from_seq([1, 2, 3]))
# print(from_seq([1]))
# print(from_seq((1,)))


def head(lst: List):
    """
    >>> head(from_seq([1, 2, 3]))
    1
    >>> head(Nil())
    Traceback (most recent call last):
    ...
    AttributeError: 'Nil' object has no attribute 'car'
    """
    return lst.car


# print(head(from_seq([1,2,3])))
# print(head(Nil()))


def tail(lst: List) -> List:
    """
    >>> tail(from_seq([1, 2, 3]))
    Cons(car=2, cdr=Cons(car=3, cdr=Nil()))
    >>> tail(from_seq([]))
    Traceback (most recent call last):
    ...
    AttributeError: 'Nil' object has no attribute 'cdr'
    """
    return lst.cdr


# print(tail(from_seq([1, 2, 3])))
# print(tail(from_seq([])))


def foldr(func: tp.Callable, acc, lst: List):
    """
    >>> foldr(lambda x, y: x + y, 0, Nil())
    0
    >>> foldr(lambda x, y: x + y, 2, from_seq([1, 2, 3]))
    8
    >>> foldr(lambda x, y: x - y, 1, from_seq([3, 2, 1]))
    1
    """
    if null(lst):
        return acc
    else:
        return func(head(lst), foldr(func, acc, tail(lst)))


# print(foldr(lambda x, y: x + y, 0, Nil()))
# print(foldr(lambda x, y: x + y, 2, from_seq([1, 2, 3])))
# print(foldr(lambda x, y: x - y, 1, from_seq([1, 2, 3])))


def foldl(func: tp.Callable, acc, lst: List):
    """
    >>> foldl(lambda x, y: x + y, 0, Nil())
    0
    >>> foldl(lambda x, y: x + y, 2, from_seq([1, 2, 3]))
    8
    >>> foldl(lambda x, y: x - y, 1, from_seq([3, 2, 1]))
    -5
    """
    if null(lst):
        return acc
    else:
        return foldl(func, func(acc, head(lst)), tail(lst))


# print(foldl(lambda x, y: x + y, 0, Nil()))
# print(foldl(lambda x, y: x + y, 2, from_seq([1, 2, 3])))
# print(foldl(lambda x, y: x - y, 1, from_seq([1, 2, 3])))


def length(lst: List) -> int:
    """
    >>> length(Nil())
    0
    >>> length(from_seq((1, 2)))
    2
    """
    if null(lst):
        return 0
    else:
        return 1 + length(tail(lst))


# print(length(from_seq(Nil())))
# print(length(from_seq((1,2))))
# print(length(from_seq((1,2,3))))


def to_list(lst: List) -> tp.List:
    """
    >>> to_list(Nil())
    []
    >>> to_list(Cons(1, Nil()))
    [1]
    >>> to_list(from_seq([1, 2, 3]))
    [1, 2, 3]
    """
    if null(lst):
        return []
    else:
        return [head(lst)] + to_list(tail(lst))


# print(to_list(Nil()))
# print(to_list(Cons(1, Nil())))
# print(to_list(from_seq([1,2,3])))


def map_(func: tp.Callable, lst: List) -> List:
    """
    >>> to_list(map_(lambda x: x, Nil()))
    []
    >>> to_list(map_(lambda x: x, from_seq([1, 2, 3])))
    [1, 2, 3]
    >>> to_list(map_(lambda x: str(x) + '0', from_seq([1, 2, 3])))
    ['10', '20', '30']
    """
    if null(lst):
        return Nil()
    else:
        # return from_seq([func(x) for x in to_list(lst)])
        return Cons(func(head(lst)), map_(func, tail(lst)))


# print(to_list(map_(lambda x: x, Nil())))
# print(to_list(map_(lambda x: x, from_seq([1, 2, 3]))))
# print(to_list(map_(lambda x: str(x) + '0', from_seq([1, 2, 3]))))


def append(lst1: List, lst2: List):
    """
    >>> append(Nil(), from_seq([]))
    Nil()
    >>> append(Nil(), Cons(0, Cons(1, Nil())))
    Cons(car=0, cdr=Cons(car=1, cdr=Nil()))
    >>> append(from_seq([1]), Nil())
    Cons(car=1, cdr=Nil())
    >>> append(from_seq([1, 2]), from_seq([3]))
    Cons(car=1, cdr=Cons(car=2, cdr=Cons(car=3, cdr=Nil())))
    """
    if null(lst1) and null(lst2):
        return Nil()
    elif null(lst1) or null(lst2):
        return lst1 if not null(lst1) else lst2
    else:
        return Cons(head(lst1), append(tail(lst1), lst2))


# print(append(Nil(), from_seq([])))
# print(append(Nil(), Cons(0, Cons(1, Nil()))))
# print(append(from_seq([1]), Nil()))
# print(append(from_seq([1, 2]), from_seq([3])))


def filter_(pred: tp.Callable, lst: List) -> List:
    """
    >>> filter_(lambda x: True, Nil())
    Nil()
    >>> to_list(filter_(lambda x: True, from_seq([1, 2])))
    [1, 2]
    >>> to_list(filter_(lambda x: False, from_seq([1, 2])))
    []
    >>> to_list(filter_(lambda x: x % 2 == 0, from_seq(range(5))))
    [0, 2, 4]
    """
    if null(lst):
        return Nil()
    if not pred(head(lst)):
        return filter_(pred, tail(lst))
    else:
        return Cons(head(lst), filter_(pred, tail(lst)))


# print(filter_(lambda x: True, Nil()))
# print(to_list(filter_(lambda x: True, from_seq([1, 2]))))
# print(to_list(filter_(lambda x: False, from_seq([1, 2]))))
# print(to_list(filter_(lambda x: x % 2 == 0, from_seq(range(5)))))


def reverse(lst: List) -> List:
    """
    >>> reverse(Nil())
    Nil()
    >>> to_list(reverse(from_seq(range(2))))
    [1, 0]
    >>> to_list(reverse(from_seq(range(3))))
    [2, 1, 0]
    """
    if null(lst):
        return Nil()
    else:
        return append(reverse(tail(lst)), Cons(head(lst), Nil()))


# print(reverse(Nil()))
# print(to_list(reverse(from_seq(range(2)))))
# print(to_list(reverse(from_seq(range(3)))))


def elem(value, lst: List) -> bool:
    """
    >>> elem(10, Nil())
    False
    >>> elem(5, from_seq(range(5)))
    False
    >>> elem(5, from_seq(range(10)))
    True
    """
    if null(lst):
        return False
    else:
        return value == head(lst) or elem(value, tail(lst))


# print(elem(10, Nil()))
# print(elem(5, from_seq(range(5))))
# print(elem(5, from_seq(range(10))))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
