import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.facets import process_facets_run
from mhs_llms.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Process FACETS score/output files into clean tables.")
    parser.add_argument(
        "--facets-dir",
        default=str(REPO_ROOT / "facets" / "human_baseline"),
        help="Directory containing FACETS score files and the FACETS output report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "human_baseline" / "facets"),
        help="Directory where processed FACETS CSV/JSON outputs should be written.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    outputs = process_facets_run(
        facets_dir=Path(args.facets_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"output_dir={outputs.output_dir}")
    print(f"combined_scores={outputs.combined_scores_path}")
    print(f"summary={outputs.summary_path}")


if __name__ == "__main__":
    main()
