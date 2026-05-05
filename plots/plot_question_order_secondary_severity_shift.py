"""Plot original versus reverse question-order severity shifts with secondary models."""

from mhs_llms.facets.order_effect_plot import (
    load_order_shift_comparison,
    load_pooled_order_delta,
    plot_order_shift_comparison,
)
from mhs_llms.paths import ARTIFACTS_DIR, DATA_DIR, FACETS_DIR
from plot_question_order_severity_shift import ORDER_SHIFT_STYLE


ORIGINAL_JUDGES_PATH = FACETS_DIR / "question_order_secondary_original_matched" / "judges_scores.csv"
REVERSE_JUDGES_PATH = FACETS_DIR / "question_order_secondary_reverse_matched" / "judges_scores.csv"
POOLED_ORDER_CONTRAST_PATH = DATA_DIR / "question_order_secondary_pooled_effect_order_contrast.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "question_order_secondary_severity_shift.png"


def main() -> None:
    """Build and save the expanded question-order severity shift figure."""

    comparison = load_order_shift_comparison(
        original_judges_path=ORIGINAL_JUDGES_PATH,
        reverse_judges_path=REVERSE_JUDGES_PATH,
    )
    pooled_delta = None
    pooled_delta_se = None
    if POOLED_ORDER_CONTRAST_PATH.exists():
        pooled_delta, pooled_delta_se = load_pooled_order_delta(POOLED_ORDER_CONTRAST_PATH)

    plotted_path = plot_order_shift_comparison(
        comparison=comparison,
        output_path=OUTPUT_PATH,
        style=ORDER_SHIFT_STYLE,
        pooled_delta=pooled_delta,
        pooled_delta_se=pooled_delta_se,
    )
    print(f"output={plotted_path.resolve()}")


if __name__ == "__main__":
    main()
