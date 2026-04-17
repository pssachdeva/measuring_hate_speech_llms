import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.config import load_llm_facets_config
from mhs_llms.facets import process_facets_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Process FACETS score/output files into clean tables.")
    parser.add_argument(
        "config",
        help="Path to a FACETS YAML config. Reads and writes processed outputs in output.facets_run_dir.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = load_llm_facets_config(Path(args.config))

    outputs = process_facets_run(
        facets_dir=config.facets_run_dir,
        output_dir=config.facets_run_dir,
    )
    print(f"facets_dir={config.facets_run_dir}")
    print(f"combined_scores={outputs.combined_scores_path}")
    print(f"summary={outputs.summary_path}")


if __name__ == "__main__":
    main()
