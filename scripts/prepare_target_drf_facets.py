import argparse
from pathlib import Path

from mhs_llms.facets.target_drf import run_target_drf_facets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FACETS files for target-identity differential rater functioning."
    )
    parser.add_argument("config_path", help="Path to the target-DRF YAML config.")
    args = parser.parse_args()

    outputs = run_target_drf_facets(Path(args.config_path))
    print(f"FACETS data: {outputs.facets_data_path}")
    print(f"FACETS spec: {outputs.facets_spec_path}")
    print(f"Target labels: {outputs.target_labels_path}")


if __name__ == "__main__":
    main()
