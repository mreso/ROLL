"""
Standalone timer utilities replacing ray.util.timer._Timer.

This module provides a _Timer class with the same API surface as Ray's
internal _Timer, allowing pipeline code to be used without a Ray dependency.
"""

import time


class _Timer:
    """A running stat for conveniently logging the duration of a code block.

    Drop-in replacement for ``ray.util.timer._Timer`` that only uses stdlib.

    Example::

        tps_timer = _Timer(window_size=5)
        with tps_timer:
            do_work()
        tps_timer.push_units_processed(n=1024)
        print(tps_timer.mean_throughput)
    """

    def __init__(self, window_size: int = 10):
        self._window_size = window_size
        self._samples: list[float] = []
        self._units_processed: list[float] = []
        self._start_time: float | None = None
        self._total_time: float = 0.0
        self.count: int = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, exc_type, exc_value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta: float):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n: float):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    def has_units_processed(self) -> bool:
        return len(self._units_processed) > 0

    @property
    def mean(self) -> float:
        if len(self._samples) == 0:
            return 0.0
        return float(sum(self._samples)) / len(self._samples)

    @property
    def mean_units_processed(self) -> float:
        if len(self._units_processed) == 0:
            return 0.0
        return float(sum(self._units_processed)) / len(self._units_processed)

    @property
    def mean_throughput(self) -> float:
        time_total = float(sum(self._samples))
        if not time_total:
            return 0.0
        return float(sum(self._units_processed)) / time_total
