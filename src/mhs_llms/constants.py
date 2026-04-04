"""Project-wide constants."""

REFERENCE_SET_PLATFORM = 1

# Human judges in the baseline FACETS export currently occupy ids up to 11142.
# Reserve a disjoint numeric block for LLM judges so linked runs never collide.
FACETS_LLM_JUDGE_ID_START = 20001

# Keep this mapping explicit so FACETS judge ids stay stable across reruns.
FACETS_LLM_JUDGE_IDS = {
    "openai:gpt-5.4": 20001,
}

# Shared response collapsing used for both the human baseline and linked LLM runs.
HUMAN_FACETS_RECODE_MAP = {
    "insult": {1: 0, 2: 1, 3: 2, 4: 3},
    "humiliate": {1: 0, 2: 0, 3: 1, 4: 2},
    "status": {1: 0, 2: 0, 3: 1, 4: 1},
    "dehumanize": {1: 0, 2: 0, 3: 1, 4: 1},
    "violence": {1: 0, 2: 0, 3: 1, 4: 1},
    "genocide": {1: 0, 2: 0, 3: 1, 4: 1},
    "attack_defend": {1: 0, 2: 1, 3: 2, 4: 3},
    "hate_speech": {1: 0, 2: 1},
}
