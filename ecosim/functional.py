from __future__ import annotations

def foldl(func, acc, seq):
    return acc if not seq else foldl(func, func(acc, seq[0]), seq[1:])


def fmap(func, seq):
    return () if not seq else (func(seq[0]),) + fmap(func, seq[1:])


def filter_f(func, seq):
    if not seq:
        return ()
    head, tail = seq[0], seq[1:]
    rest = filter_f(func, tail)
    return ((head,) + rest) if func(head) else rest
