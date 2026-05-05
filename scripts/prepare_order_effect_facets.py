import argparse
from pathlib import Path

from mhs_llms.facets import run_order_effect_facets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a pooled FACETS run for original/reverse question-order effects."
    )
    parser.add_argument("config", help="Path to the order-effect FACETS YAML config.")
    args = parser.parse_args()

    outputs = run_order_effect_facets(Path(args.config))
    print(f"facets_data={outputs.facets_data_path}")
    print(f"facets_spec={outputs.facets_spec_path}")


if __name__ == "__main__":
    main()
