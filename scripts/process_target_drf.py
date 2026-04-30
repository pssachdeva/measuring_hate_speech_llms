import argparse
from pathlib import Path

from mhs_llms.facets import process_target_drf_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a target-identity DRF FACETS report into tidy CSV tables."
    )
    parser.add_argument("config", help="Path to the target-DRF FACETS YAML config.")
    parser.add_argument(
        "--target-terms-output",
        type=Path,
        default=None,
        help="Output CSV for target terms. Defaults to data/<run>_target_terms.csv.",
    )
    parser.add_argument(
        "--pairwise-output",
        type=Path,
        default=None,
        help="Output CSV for pairwise target contrasts. Defaults to data/<run>_pairwise_contrasts.csv.",
    )
    args = parser.parse_args()

    outputs = process_target_drf_run(
        Path(args.config),
        target_terms_path=args.target_terms_output,
        pairwise_contrasts_path=args.pairwise_output,
    )
    print(f"target_terms={outputs.target_terms_path}")
    print(f"pairwise_contrasts={outputs.pairwise_contrasts_path}")


if __name__ == "__main__":
    main()
