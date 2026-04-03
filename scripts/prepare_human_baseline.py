from mhs_llms.paths import REPO_ROOT
from mhs_llms.human_baseline import run_human_baseline


if __name__ == "__main__":
    outputs = run_human_baseline(REPO_ROOT / "configs" / "human_baseline.yaml")
    print(f"facets_data={outputs.facets_data_path}")
    print(f"facets_spec={outputs.facets_spec_path}")
