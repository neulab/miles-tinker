"""Keep TorchInductor's compile-worker pool from deadlocking a trainer rank.

Inductor compiles Triton kernels in a sidecar process (``SubprocPool``) that runs a fork-context
``ProcessPoolExecutor``. With ``quiesce_async_compile_pool`` (on by default in torch 2.13) the sidecar
shuts that executor down (``shutdown(wait=False)``) after 60 s without work and the next compile forks a
fresh set of workers. When the fork lands while an old executor's manager thread holds its
``shutdown_lock`` (it holds it while joining the old workers), every new worker inherits the lock held and
blocks forever in ``concurrent.futures.process``'s ``weakref_cb`` during ``_after_fork``. The compile future
then never completes and the rank that asked for the kernel stops inside its forward/backward: CPU-blocked,
GPU idle, and invisible to the NCCL watchdog, while the other data-parallel ranks wait for it in the next
collective (the 30-minute gloo all-reduce of the Tinker executor's per-datum check).

Trainers recompile mid-run: Megatron's fused cross entropy is ``torch.compile``'d and miles calls it per
sample, so the first 1-token response a rank meets (a Tinker DP padding row) recompiles it, typically
minutes after the previous compile, just when the idle pool is being quiesced. Keeping the pool alive
removes the restart and with it the race. Only the compile workers' lifetime changes: the kernels, and so
the numerics, are the same.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

QUIESCE_ENV = "TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"
_ASYNC_COMPILE_MODULE = "torch._inductor.async_compile"


def disable_compile_pool_quiesce() -> None:
    """Turn off the idle quiesce of Inductor's compile-worker pool for this process.

    Call it early in a training process, before the first ``torch.compile``. A pool created earlier keeps
    running but loses its idle timer. A torch without the option has nothing to turn off.
    """
    os.environ[QUIESCE_ENV] = "0"  # also for any process this one starts
    # Local import: only training processes pay for importing Inductor's config, and a torch without
    # Inductor simply has nothing to turn off.
    try:
        import torch._inductor.config as inductor_config
    except ImportError:
        return
    if not hasattr(inductor_config, "quiesce_async_compile_pool"):
        return
    inductor_config.quiesce_async_compile_pool = False
    _stop_running_pool_timer()
    logger.info("Inductor compile-worker pool: idle quiesce disabled")


def _stop_running_pool_timer() -> None:
    # The pool only exists once async_compile is imported; never import it here (it is heavy).
    async_compile = sys.modules.get(_ASYNC_COMPILE_MODULE)
    process_pool = getattr(getattr(async_compile, "AsyncCompile", None), "process_pool", None)
    cache_info = getattr(process_pool, "cache_info", None)
    if cache_info is None or not cache_info().currsize:
        return
    pool = process_pool()
    timer = getattr(pool, "timer", None)
    if timer is not None:
        timer.quit()
        pool.timer = None
        logger.info("Inductor compile-worker pool was already running: stopped its idle quiesce timer")
