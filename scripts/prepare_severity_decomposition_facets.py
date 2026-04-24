import argparse
from pathlib import Path

from mhs_llms.facets import run_severity_decomposition_facets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a FACETS run for LLM item-dependent severity decomposition."
    )
    parser.add_argument("config", help="Path to the severity decomposition FACETS YAML config.")
    args = parser.parse_args()

    outputs = run_severity_decomposition_facets(Path(args.config))
    print(f"facets_data={outputs.facets_data_path}")
    print(f"facets_spec={outputs.facets_spec_path}")


if __name__ == "__main__":
    main()
