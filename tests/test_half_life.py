import numpy as np
import pandas as pd
import pytest

from quant_research_toolkit.time_series.half_life import estimate_half_life


def generate_ar1_series(phi: float, n: int = 3000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)

    x = np.zeros(n)

    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]

    return pd.Series(x)


def test_ar1_mean_reverting_series_has_positive_half_life():
    series = generate_ar1_series(phi=0.8)

    result = estimate_half_life(series)

    assert result.is_mean_reverting is True
    assert result.lambda_ < 0
    assert 1.0 < result.half_life < 10.0


def test_stronger_mean_reversion_has_shorter_half_life():
    fast_series = generate_ar1_series(phi=0.5)
    slow_series = generate_ar1_series(phi=0.95)

    fast_result = estimate_half_life(fast_series)
    slow_result = estimate_half_life(slow_series)

    assert fast_result.is_mean_reverting is True
    assert slow_result.is_mean_reverting is True
    assert fast_result.half_life < slow_result.half_life


def test_half_life_rejects_too_short_series():
    short_series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        estimate_half_life(short_series)
