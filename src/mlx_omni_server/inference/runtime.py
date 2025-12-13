import asyncio
import threading
import weakref
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from fastapi.concurrency import run_in_threadpool

P = ParamSpec("P")
R = TypeVar("R")


_mlx_gate_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)
_mlx_gate_lock = threading.Lock()


def get_mlx_gate() -> asyncio.Lock:
    """Return the shared MLX gate for the current event loop.

    The server normally runs a single event loop, but tests may create multiple loops.
    """
    loop = asyncio.get_running_loop()
    with _mlx_gate_lock:
        gate = _mlx_gate_by_loop.get(loop)
        if gate is None:
            gate = asyncio.Lock()
            _mlx_gate_by_loop[loop] = gate
    return gate


async def run_blocking(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a blocking function in the threadpool (no MLX gate)."""
    return await run_in_threadpool(func, *args, **kwargs)


async def run_mlx(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run MLX-backed work under a shared gate and in the threadpool."""
    async with get_mlx_gate():
        return await run_in_threadpool(func, *args, **kwargs)
