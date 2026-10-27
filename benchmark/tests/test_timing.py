import pytest
from skala_benchmark.timing import CudaTimeline, HostTimeline, _CudaMark


class _FakeCudaMark:
    def __init__(self, elapsed_ms: float) -> None:
        self.elapsed_ms = elapsed_ms

    def elapsed_time(self, end_event: _CudaMark) -> float:
        return self.elapsed_ms


def test_host_timeline_requires_host_marks() -> None:
    timeline = HostTimeline()

    assert timeline.elapsed_ms(1.0, 1.25) == pytest.approx(250.0)
    with pytest.raises(TypeError, match="requires host-clock marks"):
        timeline.elapsed_ms(_FakeCudaMark(1.0), 1.25)


def test_cuda_timeline_requires_cuda_marks() -> None:
    timeline = CudaTimeline()
    start = _FakeCudaMark(2.5)
    end = _FakeCudaMark(0.0)

    assert timeline.elapsed_ms(start, end) == pytest.approx(2.5)
    with pytest.raises(TypeError, match="requires CUDA event marks"):
        timeline.elapsed_ms(start, 1.25)
