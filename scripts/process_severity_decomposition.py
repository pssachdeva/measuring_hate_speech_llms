import argparse
from pathlib import Path

from mhs_llms.facets import process_severity_decomposition_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a severity decomposition FACETS report into a tidy bias-term dataset."
    )
    parser.add_argument("config", help="Path to the severity decomposition FACETS YAML config.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to data/<facets_run_dir_name>_bias_terms.csv.",
    )
    args = parser.parse_args()

    outputs = process_severity_decomposition_run(Path(args.config), output_path=args.output)
    print(f"bias_terms={outputs.bias_terms_path}")


if __name__ == "__main__":
    main()
