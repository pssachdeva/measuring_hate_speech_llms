"""Shared repository path helpers."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"
FACETS_DIR = REPO_ROOT / "facets"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIGS_DIR = REPO_ROOT / "configs"
