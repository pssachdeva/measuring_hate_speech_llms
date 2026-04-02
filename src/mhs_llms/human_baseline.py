"""Human baseline generation for FACETS exports."""

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

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
    run_dir: Path
    comment_ids_path: Path
    cleaned_annotations_path: Path
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

    # Read the raw MHS data for the configured split.
    dataset = load_mhs_dataframe(
        dataset_name=config.dataset.name,
        split=config.dataset.split,
        config_name=config.dataset.config_name,
    )

    # Convert raw HF-coded human annotations into the repo's normalized schema.
    normalized_annotations = normalize_human_annotations(dataset)

    # Create the output directories for tabular artifacts and FACETS control files.
    run_dir = config.output.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    facets_run_dir = config.output.facets_run_dir
    facets_run_dir.mkdir(parents=True, exist_ok=True)

    # Write the deduplicated comment ids so the evaluated comment set is explicit.
    comment_ids_path = run_dir / config.output.comment_ids_filename
    comment_ids = tuple(sorted(dataset["comment_id"].drop_duplicates().astype(int).tolist()))
    pd.DataFrame({"comment_id": comment_ids}).to_csv(comment_ids_path, index=False)

    # Persist the cleaned annotation table for inspection and reuse.
    cleaned_annotations_path = run_dir / config.output.cleaned_annotations_filename
    normalized_annotations.to_csv(cleaned_annotations_path, index=False)

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
    facets_data_path = run_dir / config.output.facets_data_filename
    write_facets_data(facets_frame, facets_data_path)

    # Build the FACETS spec alongside a relative reference to the data file so the
    # generated spec can be executed from the dedicated facets/ directory.
    facets_spec_path = facets_run_dir / config.output.facets_spec_filename
    facets_data_reference = Path(os.path.relpath(facets_data_path, start=facets_spec_path.parent))
    spec_text = build_facets_spec(
        facets_frame=facets_frame,
        facets_config=config.facets,
        data_filename=str(facets_data_reference),
        score_filename=config.output.facets_score_filename,
        output_filename=config.output.facets_output_filename,
    )
    write_facets_spec(spec_text, facets_spec_path)

    return HumanBaselineOutputs(
        run_dir=run_dir,
        comment_ids_path=comment_ids_path,
        cleaned_annotations_path=cleaned_annotations_path,
        facets_data_path=facets_data_path,
        facets_spec_path=facets_spec_path,
    )
