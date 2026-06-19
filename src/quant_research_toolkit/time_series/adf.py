from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class ADFResult:
    test_statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Dict[str, float]
    n_lags: int
    n_observations: int


def adf_test(
    series: pd.Series | np.ndarray,
    significance_level: float = 0.05,
    autolag: str = "AIC",
) -> ADFResult:
    """
    Run the Augmented Dickey-Fuller test.

    Null hypothesis:
        The series has a unit root and is likely non-stationary.

    Alternative hypothesis:
        The series does not have a unit root and is more likely stationary.

    Interpretation:
        If p_value < significance_level, reject the null hypothesis.
        The series is treated as stationary.
    """
    clean_series = pd.Series(series).dropna()

    if len(clean_series) < 20:
        raise ValueError("ADF test requires at least 20 non-null observations.")

    result = adfuller(clean_series, autolag=autolag)

    test_statistic = float(result[0])
    p_value = float(result[1])
    n_lags = int(result[2])
    n_observations = int(result[3])
    critical_values = {key: float(value) for key, value in result[4].items()}

    return ADFResult(
        test_statistic=test_statistic,
        p_value=p_value,
        is_stationary=p_value < significance_level,
        critical_values=critical_values,
        n_lags=n_lags,
        n_observations=n_observations,
    )
