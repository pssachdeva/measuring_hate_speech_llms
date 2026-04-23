"""Plot item-level annotator agreement for reference-set LLM and human ratings."""

from cycler import cycler
import matplotlib.pyplot as plt
from mpl_lego.style import use_latex_style

from mhs_llms.annotator_agreement import (
    build_item_agreement_summary,
    load_annotation_files,
    load_human_annotations,
    plot_item_agreement_summary,
    reference_comment_ids_from_annotations,
)
from mhs_llms.paths import ARTIFACTS_DIR, DATA_DIR, REPO_ROOT
from mhs_llms.schema import ITEM_NAMES


MODEL_ANNOTATION_PATHS = [
    DATA_DIR / "reference_set_openai_processed.csv",
    DATA_DIR / "reference_set_anthropic_processed.csv",
    DATA_DIR / "reference_set_google_processed.csv",
    DATA_DIR / "reference_set_xai_processed.csv",
    DATA_DIR / "reference_set_open_large_processed.csv",
]
HUMAN_ANNOTATION_PATH = None
REFERENCE_SET_PATH = None
OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_annotator_agreement.png"
SUMMARY_OUTPUT_PATH = DATA_DIR / "reference_set_annotator_agreement.csv"
DATASET_NAME = "ucberkeley-dlab/measuring-hate-speech"
DATASET_SPLIT = "train"
DATASET_CONFIG_NAME = None
ITEMS_TO_PLOT = ITEM_NAMES
DISTANCE_METRIC = "ordinal"
FIGSIZE = (8.2, 4.4)
DPI = 300
MARKER_SIZE = 62.0
X_OFFSET = 0.18
Y_LIMITS = (-0.05, 1.0)
TICK_LABEL_SIZE = 8.0
AXIS_LABEL_SIZE = 10.0
LEGEND_FONT_SIZE = 8.5
COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
]


def main() -> None:
    """Build and save the reference-set item agreement figure."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    model_annotations = load_annotation_files(MODEL_ANNOTATION_PATHS)
    comment_ids = reference_comment_ids_from_annotations(
        llm_annotations=model_annotations,
        reference_set_path=REFERENCE_SET_PATH,
    )
    human_annotations = load_human_annotations(
        human_path=HUMAN_ANNOTATION_PATH,
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        config_name=DATASET_CONFIG_NAME,
    )

    agreement_summary = build_item_agreement_summary(
        llm_annotations=model_annotations,
        human_annotations=human_annotations,
        reference_comment_ids=comment_ids,
        item_names=ITEMS_TO_PLOT,
        distance_metric=DISTANCE_METRIC,
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agreement_summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    plotted_path = plot_item_agreement_summary(
        summary_frame=agreement_summary,
        output_path=OUTPUT_PATH,
        figsize=FIGSIZE,
        dpi=DPI,
        marker_size=MARKER_SIZE,
        x_offset=X_OFFSET,
        y_limits=Y_LIMITS,
        tick_label_size=TICK_LABEL_SIZE,
        axis_label_size=AXIS_LABEL_SIZE,
        legend_font_size=LEGEND_FONT_SIZE,
    )

    print(f"repo={REPO_ROOT}")
    print(f"model_files={','.join(str(path) for path in MODEL_ANNOTATION_PATHS)}")
    print(f"num_reference_comments={len(comment_ids)}")
    print(f"distance_metric={DISTANCE_METRIC}")
    print(f"summary={SUMMARY_OUTPUT_PATH}")
    print(f"output={plotted_path}")


if __name__ == "__main__":
    main()
