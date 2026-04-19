"""Plot reference-set model average hate scores against human averages."""

from cycler import cycler
import matplotlib.pyplot as plt
from mpl_lego.style import use_latex_style

from mhs_llms.hate_score_figure import (
    build_human_average_scores,
    build_model_comment_scores,
    load_model_annotation_files,
    plot_average_hate_scores,
    reference_comment_ids,
    summarize_model_scores,
)
from mhs_llms.paths import ARTIFACTS_DIR, REPO_ROOT


MODEL_ANNOTATION_PATHS = [
    REPO_ROOT / "data" / "reference_set_openai_processed.csv",
    REPO_ROOT / "data" / "reference_set_anthropic_processed.csv",
    REPO_ROOT / "data" / "reference_set_google_processed.csv",
    REPO_ROOT / "data" / "reference_set_xai_processed.csv",
    REPO_ROOT / "data" / "reference_set_open_large_processed.csv",
]
OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_average_hate_scores.png"
DATASET_NAME = "ucberkeley-dlab/measuring-hate-speech"
DATASET_SPLIT = "train"
DATASET_CONFIG_NAME = None
N_BOOTSTRAP = 10000
RANDOM_SEED = 20260418
PROVIDER_GAP = 1.75
FIGSIZE = (16.0, 7.2)
DPI = 300
SCORE_MIN = 0.0
SCORE_MAX = 38.0
KDE_POINTS = 512
MARKER_SIZE = 5.2
ERRORBAR_LINEWIDTH = 1.15
CAPSIZE = 2.4
TICK_LABEL_SIZE = 6.5
AXIS_LABEL_SIZE = 10.0
PROVIDER_LABEL_SIZE = 8.5
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
    """Build and save the average hate-score comparison figure."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    model_annotations = load_model_annotation_files(MODEL_ANNOTATION_PATHS)
    comment_ids = reference_comment_ids(model_annotations)
    model_comment_scores = build_model_comment_scores(model_annotations)
    model_summary = summarize_model_scores(
        model_comment_scores=model_comment_scores,
        n_bootstrap=N_BOOTSTRAP,
        random_seed=RANDOM_SEED,
    )
    human_average_scores = build_human_average_scores(
        comment_ids=comment_ids,
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        config_name=DATASET_CONFIG_NAME,
    )

    plotted_path = plot_average_hate_scores(
        model_summary=model_summary,
        human_average_scores=human_average_scores,
        output_path=OUTPUT_PATH,
        provider_gap=PROVIDER_GAP,
        figsize=FIGSIZE,
        dpi=DPI,
        score_min=SCORE_MIN,
        score_max=SCORE_MAX,
        kde_points=KDE_POINTS,
        marker_size=MARKER_SIZE,
        errorbar_linewidth=ERRORBAR_LINEWIDTH,
        capsize=CAPSIZE,
        tick_label_size=TICK_LABEL_SIZE,
        axis_label_size=AXIS_LABEL_SIZE,
        provider_label_size=PROVIDER_LABEL_SIZE,
    )

    print(f"model_files={','.join(str(path) for path in MODEL_ANNOTATION_PATHS)}")
    print(f"num_reference_comments={len(comment_ids)}")
    print(f"num_model_runs={len(model_summary)}")
    print(f"num_human_annotators={len(human_average_scores)}")
    print(f"output={plotted_path}")


if __name__ == "__main__":
    main()
