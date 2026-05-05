import argparse
from pathlib import Path

from mhs_llms.facets import process_order_effect_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process pooled order-effect FACETS scores into tidy CSV tables."
    )
    parser.add_argument("config", help="Path to the order-effect FACETS YAML config.")
    parser.add_argument(
        "--order-conditions-output",
        type=Path,
        default=None,
        help="Output CSV for order-condition scores. Defaults to data/<run>_order_conditions.csv.",
    )
    parser.add_argument(
        "--order-contrast-output",
        type=Path,
        default=None,
        help="Output CSV for reverse-minus-original contrast. Defaults to data/<run>_order_contrast.csv.",
    )
    args = parser.parse_args()

    outputs = process_order_effect_run(
        Path(args.config),
        order_conditions_path=args.order_conditions_output,
        order_contrast_path=args.order_contrast_output,
    )
    print(f"order_conditions={outputs.order_conditions_path}")
    print(f"order_contrast={outputs.order_contrast_path}")


if __name__ == "__main__":
    main()
