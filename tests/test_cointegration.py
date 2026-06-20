import numpy as np
import pandas as pd
import pytest

from quant_research_toolkit.pairs.cointegration import cointegration_test
from quant_research_toolkit.pairs.hedge_ratio import estimate_hedge_ratio


def generate_random_walk(n: int = 2000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 1.0, n)
    return pd.Series(np.cumsum(shocks))


def generate_cointegrated_pair(
    n: int = 2000,
    hedge_ratio: float = 2.0,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)

    x = generate_random_walk(n=n, seed=seed)

    stationary_noise = rng.normal(0.0, 1.0, n)

    y = hedge_ratio * x + stationary_noise

    return pd.Series(y), pd.Series(x)


def generate_correlated_but_not_cointegrated_pair(
    n: int = 2000,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Construct two highly correlated non-stationary series.

    y = x + 0.2 * z

    where x and z are independent random walks.

    y and x can be highly correlated, but their spread contains another
    random walk component, so the spread should not be stationary.
    """
    x = generate_random_walk(n=n, seed=seed)
    z = generate_random_walk(n=n, seed=seed + 1)

    y = x + 0.2 * z

    return pd.Series(y), pd.Series(x)


def test_estimate_hedge_ratio_recovers_known_beta():
    y, x = generate_cointegrated_pair(hedge_ratio=2.0)

    result = estimate_hedge_ratio(y=y, x=x)

    assert result.hedge_ratio == pytest.approx(2.0, abs=0.05)
    assert len(result.spread) == len(x)


def test_cointegrated_pair_has_stationary_spread():
    y, x = generate_cointegrated_pair(hedge_ratio=2.0)

    result = cointegration_test(y=y, x=x)

    assert result.is_cointegrated is True
    assert result.spread_adf_result.p_value < 0.05


def test_highly_correlated_pair_can_fail_cointegration_test():
    y, x = generate_correlated_but_not_cointegrated_pair()

    correlation = np.corrcoef(y, x)[0, 1]
    result = cointegration_test(y=y, x=x)

    assert correlation > 0.8
    assert result.is_cointegrated is False
    assert result.spread_adf_result.p_value >= 0.05


def test_cointegration_rejects_too_short_series():
    y = pd.Series([1.0, 2.0, 3.0])
    x = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        cointegration_test(y=y, x=x)
