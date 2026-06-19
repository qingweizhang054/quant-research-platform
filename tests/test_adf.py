import numpy as np
import pandas as pd
import pytest

from quant_research_toolkit.time_series.adf import adf_test


def test_white_noise_is_stationary():
    rng = np.random.default_rng(seed=42)
    white_noise = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))

    result = adf_test(white_noise)

    assert result.is_stationary is True
    assert result.p_value < 0.05


def test_random_walk_is_non_stationary():
    rng = np.random.default_rng(seed=42)
    shocks = rng.normal(loc=0.0, scale=1.0, size=2000)
    random_walk = pd.Series(np.cumsum(shocks))

    result = adf_test(random_walk)

    assert result.is_stationary is False
    assert result.p_value >= 0.05


def test_adf_rejects_too_short_series():
    short_series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        adf_test(short_series)
