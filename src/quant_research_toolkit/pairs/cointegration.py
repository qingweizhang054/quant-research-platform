from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_toolkit.pairs.hedge_ratio import (
    HedgeRatioResult,
    estimate_hedge_ratio,
)
from quant_research_toolkit.time_series.adf import ADFResult, adf_test


@dataclass(frozen=True)
class CointegrationResult:
    is_cointegrated: bool
    hedge_ratio_result: HedgeRatioResult
    spread_adf_result: ADFResult


def cointegration_test(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    significance_level: float = 0.05,
) -> CointegrationResult:
    """
    Test whether two series are cointegrated using the Engle-Granger approach.

    Step 1:
        Regress y on x:
            y_t = alpha + beta * x_t + residual_t

    Step 2:
        Run ADF test on the residual / spread.

    If the spread is stationary, y and x are treated as cointegrated.

    Important:
        This is not the Johansen test. This is a simple two-series
        cointegration test suitable for pairs trading research.
    """
    hedge_ratio_result = estimate_hedge_ratio(y=y, x=x)

    spread_adf_result = adf_test(
        hedge_ratio_result.spread,
        significance_level=significance_level,
    )

    return CointegrationResult(
        is_cointegrated=spread_adf_result.is_stationary,
        hedge_ratio_result=hedge_ratio_result,
        spread_adf_result=spread_adf_result,
    )
