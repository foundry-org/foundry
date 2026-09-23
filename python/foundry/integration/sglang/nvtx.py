# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""NVTX phase ranges for the SGLang integration, so an Nsight Systems trace of a SAVE or LOAD
engine attributes CUDA API time to foundry's phases (binary restore, memory restore, graph
capture / save / restore). No-op cost when no profiler is attached."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

F = TypeVar("F", bound=Callable)


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_traced(name: str) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with nvtx_range(name):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco
