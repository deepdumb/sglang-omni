"""Local profiling aid: lets a stage's own scheduler thread start/stop the torch profiler,
so CPU operator events (thread-local in torch) are recorded for the thread doing the work."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
TARGET_STAGE = os.environ.get("SGLANG_OMNI_PROFILE_THREAD_STAGE")
PENDING: tuple | None = None


def poll() -> None:
    global PENDING
    cmd = PENDING
    if cmd is None:
        return
    PENDING = None
    from .torch_profiler import TorchProfiler

    if cmd[0] == "start":
        TorchProfiler.start(cmd[1], run_id=cmd[2])
        logger.info("thread_hook: profiler started on scheduler thread")
    else:
        TorchProfiler.stop(run_id=cmd[1])
        logger.info("thread_hook: profiler stopped on scheduler thread")
