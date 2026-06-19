from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeRatioResult:
    intercept: float
    hedge_ratio: float
    spread: pd.Series


def estimate_hedge_ratio(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
) -> HedgeRatioResult:
    """
    Estimate hedge ratio using OLS.

    Model:
        y_t = intercept + beta * x_t + residual_t

    The spread is:
        spread_t = y_t - intercept - beta * x_t

    In pairs trading, beta is the hedge ratio.
    """
    data = pd.concat(
        [
            pd.Series(y, name="y"),
            pd.Series(x, name="x"),
        ],
        axis=1,
    ).dropna()

    if len(data) < 30:
        raise ValueError("Hedge ratio estimation requires at least 30 observations.")

    y_values = data["y"].to_numpy(dtype=float)
    x_values = data["x"].to_numpy(dtype=float)

    design_matrix = np.column_stack(
        [
            np.ones(len(x_values)),
            x_values,
        ]
    )

    beta = np.linalg.lstsq(design_matrix, y_values, rcond=None)[0]

    intercept = float(beta[0])
    hedge_ratio = float(beta[1])

    spread_values = y_values - intercept - hedge_ratio * x_values
    spread = pd.Series(spread_values, index=data.index, name="spread")

    return HedgeRatioResult(
        intercept=intercept,
        hedge_ratio=hedge_ratio,
        spread=spread,
    )
