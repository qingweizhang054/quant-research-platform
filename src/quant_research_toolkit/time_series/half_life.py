from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HalfLifeResult:
    half_life: float
    lambda_: float
    is_mean_reverting: bool


def estimate_half_life(series: pd.Series | np.ndarray) -> HalfLifeResult:
    """
    Estimate the half-life of mean reversion.

    Model:
        Δy_t = alpha + lambda * y_{t-1} + error_t

    If lambda < 0, the series is mean-reverting.

    Half-life:
        -log(2) / lambda

    Interpretation:
        Half-life is the approximate number of periods required for a deviation
        from the mean to decay by 50%.
    """
    y = pd.Series(series).dropna().astype(float)

    if len(y) < 30:
        raise ValueError("Half-life estimation requires at least 30 observations.")

    y_lag = y.shift(1).dropna()
    delta_y = y.diff().dropna()

    # Align indices
    y_lag = y_lag.loc[delta_y.index]

    # Regression: delta_y = alpha + lambda * y_lag + error
    x = np.column_stack([np.ones(len(y_lag)), y_lag.to_numpy()])
    beta = np.linalg.lstsq(x, delta_y.to_numpy(), rcond=None)[0]

    lambda_ = float(beta[1])

    if lambda_ >= 0:
        return HalfLifeResult(
            half_life=np.inf,
            lambda_=lambda_,
            is_mean_reverting=False,
        )

    half_life = float(-np.log(2) / lambda_)

    return HalfLifeResult(
        half_life=half_life,
        lambda_=lambda_,
        is_mean_reverting=True,
    )
