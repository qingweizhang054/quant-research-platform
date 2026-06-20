from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running this file directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from quant_research_toolkit.pairs.cointegration import cointegration_test
from quant_research_toolkit.pairs.hedge_ratio import estimate_hedge_ratio
from quant_research_toolkit.time_series.adf import adf_test
from quant_research_toolkit.time_series.half_life import estimate_half_life


FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def generate_white_noise(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 1.0, n), name="white_noise")


def generate_random_walk(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 1.0, n)
    return pd.Series(np.cumsum(shocks), name="random_walk")


def generate_ar1(phi: float = 0.8, n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)

    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]

    return pd.Series(x, name=f"ar1_phi_{phi}")


def generate_cointegrated_pair(
    n: int = 1000,
    hedge_ratio: float = 2.0,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)

    x = generate_random_walk(n=n, seed=seed)
    stationary_noise = pd.Series(rng.normal(0.0, 1.0, n))

    y = hedge_ratio * x + stationary_noise

    x.name = "x_random_walk"
    y.name = "y_cointegrated"

    return y, x


def save_plot(series: pd.Series, title: str, filename: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close()


def main() -> None:
    white_noise = generate_white_noise()
    random_walk = generate_random_walk()
    ar1 = generate_ar1(phi=0.8)
    y, x = generate_cointegrated_pair()

    white_noise_adf = adf_test(white_noise)
    random_walk_adf = adf_test(random_walk)
    ar1_adf = adf_test(ar1)
    ar1_half_life = estimate_half_life(ar1)

    hedge_result = estimate_hedge_ratio(y=y, x=x)
    coint_result = cointegration_test(y=y, x=x)

    print("=== Chapter 2 Mean Reversion Demo ===")
    print()
    print("White noise:")
    print(f"  ADF p-value: {white_noise_adf.p_value:.6f}")
    print(f"  Stationary: {white_noise_adf.is_stationary}")
    print()
    print("Random walk:")
    print(f"  ADF p-value: {random_walk_adf.p_value:.6f}")
    print(f"  Stationary: {random_walk_adf.is_stationary}")
    print()
    print("AR(1) mean-reverting series:")
    print(f"  ADF p-value: {ar1_adf.p_value:.6f}")
    print(f"  Stationary: {ar1_adf.is_stationary}")
    print(f"  Estimated half-life: {ar1_half_life.half_life:.2f} periods")
    print()
    print("Cointegrated pair:")
    print(f"  Estimated hedge ratio: {hedge_result.hedge_ratio:.4f}")
    print(f"  Spread ADF p-value: {coint_result.spread_adf_result.p_value:.6f}")
    print(f"  Cointegrated: {coint_result.is_cointegrated}")

    save_plot(
        white_noise,
        "White Noise: Stationary Series",
        "01_white_noise_stationary.png",
    )

    save_plot(
        random_walk,
        "Random Walk: Non-Stationary Series",
        "02_random_walk_non_stationary.png",
    )

    save_plot(
        ar1,
        "AR(1): Mean-Reverting Series",
        "03_ar1_mean_reverting.png",
    )

    save_plot(
        hedge_result.spread,
        "Cointegrated Spread: Stationary Residual",
        "04_cointegrated_spread.png",
    )


if __name__ == "__main__":
    main()
