import argparse
from pathlib import Path

from mhs_llms.llm_facets import run_anchored_llm_facets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an LLM-only FACETS run linked to human anchors.")
    parser.add_argument("config", help="Path to the linked LLM FACETS YAML config.")
    args = parser.parse_args()

    outputs = run_anchored_llm_facets(Path(args.config))
    print(f"facets_data={outputs.facets_data_path}")
    print(f"facets_spec={outputs.facets_spec_path}")


if __name__ == "__main__":
    main()
