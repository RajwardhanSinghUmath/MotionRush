"""One high-resolution monotonic receipt clock shared across processes."""

import time


def now_ns():
    # Windows perf_counter is cross-process from supported Python 3.10 onward.
    # Python 3.10 monotonic may use the coarser GetTickCount64 clock.
    return time.perf_counter_ns()
