import argparse
from pathlib import Path

from mhs_llms.facets.llm_only import run_llm_only_facets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an unanchored LLM-only FACETS run.")
    parser.add_argument("config", help="Path to the unanchored LLM-only FACETS YAML config.")
    args = parser.parse_args()

    outputs = run_llm_only_facets(Path(args.config))
    print(f"facets_data={outputs.facets_data_path}")
    print(f"facets_spec={outputs.facets_spec_path}")


if __name__ == "__main__":
    main()
