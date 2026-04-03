"""Human baseline generation for FACETS exports."""

from dataclasses import dataclass
from pathlib import Path

from mhs_llms.config import load_human_baseline_config
from mhs_llms.dataset import load_mhs_dataframe
from mhs_llms.facets import (
    build_facets_spec,
    build_human_facets_frame,
    write_facets_data,
    write_facets_spec,
)
from mhs_llms.schema import normalize_human_annotations
from mhs_llms.utils import recode_responses


@dataclass(frozen=True)
class HumanBaselineOutputs:
    facets_data_path: Path
    facets_spec_path: Path


def run_human_baseline(config_path: Path) -> HumanBaselineOutputs:
    """Generate the human-only baseline artifacts used to run FACETS.

    The pipeline loads the configured MHS dataset split, normalizes the full
    human annotation table into the internal format, and writes the outputs
    needed for downstream FACETS analysis.
    """

    # Load the run configuration so dataset settings and output locations are explicit.
    config = load_human_baseline_config(config_path)

    # Read the raw MHS train split from the fixed dataset source.
    dataset = load_mhs_dataframe(
        dataset_name="ucberkeley-dlab/measuring-hate-speech",
        split="train",
        config_name=None,
    )

    # Convert raw HF-coded human annotations into the repo's normalized schema.
    normalized_annotations = normalize_human_annotations(dataset)

    # Create the output directory for FACETS control files and data.
    facets_run_dir = config.output.facets_run_dir
    facets_run_dir.mkdir(parents=True, exist_ok=True)

    # Recode the numeric survey responses into the collapsed categories used for FACETS.
    facets_annotations = recode_responses(
        normalized_annotations,
        insult={1: 0, 2: 1, 3: 2, 4: 3},
        humiliate={1: 0, 2: 0, 3: 1, 4: 2},
        status={1: 0, 2: 0, 3: 1, 4: 1},
        dehumanize={1: 0, 2: 0, 3: 1, 4: 1},
        violence={1: 0, 2: 0, 3: 1, 4: 1},
        genocide={1: 0, 2: 0, 3: 1, 4: 1},
        attack_defend={1: 0, 2: 1, 3: 2, 4: 3},
        hate_speech={1: 0, 2: 1},
    )

    # Reshape the recoded annotations into FACETS input format and write the TSV.
    facets_frame = build_human_facets_frame(facets_annotations)
    facets_data_path = facets_run_dir / config.output.facets_data_filename
    write_facets_data(facets_frame, facets_data_path)

    # Build the FACETS spec so it points at the co-located FACETS data file.
    facets_spec_path = facets_run_dir / config.output.facets_spec_filename
    spec_text = build_facets_spec(
        facets_frame=facets_frame,
        facets_config=config.facets,
        data_filename=config.output.facets_data_filename,
        score_filename=config.output.facets_score_filename,
        output_filename=config.output.facets_output_filename,
    )
    write_facets_spec(spec_text, facets_spec_path)

    return HumanBaselineOutputs(
        facets_data_path=facets_data_path,
        facets_spec_path=facets_spec_path,
    )
