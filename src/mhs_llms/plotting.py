"""Shared plotting helpers used across repository figures."""

from pathlib import Path
import math

import matplotlib.figure
from mpl_lego.labels import bold_text
from mpl_lego.style import check_latex_style_on, use_latex_style
import pandas as pd


PROVIDER_COLORS = {
    "anthropic": "#4C9CC4",
    "openai": "#51B26D",
    "google": "#5E8DE0",
    "deepseek": "#9A63C5",
    "xai": "#E24A63",
    "unknown": "#7A7A7A",
}


def apply_plot_style() -> None:
    """Enable the shared mpl_lego plotting style."""

    use_latex_style()


def format_plot_text(text: str | list[str]) -> str | list[str]:
    """Use mpl_lego bold labels only when the LaTeX style is active."""

    if check_latex_style_on():
        return bold_text(text)
    return text


def get_provider_color(provider_name: str) -> str:
    """Return the configured color for one provider slug."""

    return PROVIDER_COLORS.get(provider_name, PROVIDER_COLORS["unknown"])


def save_figure(figure: matplotlib.figure.Figure, output_path: Path, dpi: int = 300) -> Path:
    """Write one figure to disk and return the resolved output path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def build_gaussian_kde_curve(
    values: list[float],
    x_min: float,
    x_max: float,
    point_count: int = 512,
) -> tuple[list[float], list[float]]:
    """Estimate a smooth Gaussian KDE curve for a list of scalar values."""

    if not values:
        raise ValueError("KDE requires at least one score")

    bandwidth = _estimate_bandwidth(values)
    x_values = [x_min + (x_max - x_min) * index / max(point_count - 1, 1) for index in range(point_count)]
    normalizer = len(values) * bandwidth * math.sqrt(2.0 * math.pi)
    y_values: list[float] = []

    for x_value in x_values:
        kernel_sum = 0.0
        for sample in values:
            standardized = (x_value - sample) / bandwidth
            kernel_sum += math.exp(-0.5 * standardized * standardized)
        y_values.append(kernel_sum / normalizer)

    return x_values, y_values


def _estimate_bandwidth(values: list[float]) -> float:
    """Estimate a KDE bandwidth with a Silverman-style fallback."""

    if len(values) == 1:
        return 1.0

    series = pd.Series(values, dtype=float)
    standard_deviation = float(series.std(ddof=1))
    interquartile_range = float(series.quantile(0.75) - series.quantile(0.25))
    scaled_iqr = interquartile_range / 1.34 if interquartile_range else 0.0
    positive_spreads = [value for value in (standard_deviation, scaled_iqr) if value > 0.0]
    spread = min(positive_spreads) if positive_spreads else standard_deviation
    if spread <= 0.0:
        return 1.0
    return max(0.3, 0.9 * spread * (len(values) ** (-1.0 / 5.0)))
