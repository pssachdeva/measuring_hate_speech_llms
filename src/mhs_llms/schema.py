"""Internal MHS annotation schema shared by human and future model outputs."""

from dataclasses import dataclass, field
import json
from typing import Any

import pandas as pd


ITEM_NAMES = (
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
    "hate_speech",
)

ITEM_RESPONSE_LETTERS = {
    "sentiment": ("A", "B", "C", "D", "E"),
    "respect": ("A", "B", "C", "D", "E"),
    "insult": ("A", "B", "C", "D", "E"),
    "humiliate": ("A", "B", "C", "D", "E"),
    "status": ("A", "B", "C", "D", "E"),
    "dehumanize": ("A", "B", "C", "D", "E"),
    "violence": ("A", "B", "C", "D", "E"),
    "genocide": ("A", "B", "C", "D", "E"),
    "attack_defend": ("A", "B", "C", "D", "E"),
    "hate_speech": ("A", "B", "C"),
}

TARGET_GROUP_CODES = ("A", "B", "C", "D", "E", "F", "G", "H", "I")

# These mappings normalize human HF-coded responses into prompt-order letters so
# future model annotations can share the same external response space.
HF_VALUE_TO_PROMPT_LETTER = {
    "sentiment": {0: "E", 1: "D", 2: "C", 3: "B", 4: "A"},
    "respect": {0: "E", 1: "D", 2: "C", 3: "B", 4: "A"},
    "insult": {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
    "humiliate": {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
    "status": {0: "E", 1: "D", 2: "C", 3: "B", 4: "A"},
    "dehumanize": {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
    "violence": {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
    "genocide": {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
    "attack_defend": {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"},
    "hate_speech": {0: "B", 1: "C", 2: "A"},
}

PROMPT_LETTER_TO_HF_VALUE = {
    item_name: {letter: value for value, letter in value_to_letter.items()}
    for item_name, value_to_letter in HF_VALUE_TO_PROMPT_LETTER.items()
}

TARGET_GROUP_COLUMNS = {
    "A": {
        "target_race_black": "Black or African American",
        "target_race_latinx": "Latino or non-white Hispanic",
        "target_race_asian": "Asian",
        "target_race_middle_eastern": "Middle Eastern",
        "target_race_native_american": "Native American or Alaska Native",
        "target_race_pacific_islander": "Pacific Islander",
        "target_race_white": "Non-Hispanic white",
        "target_race_other": "Other race or ethnicity",
    },
    "B": {
        "target_religion_jewish": "Jews",
        "target_religion_christian": "Christians",
        "target_religion_buddhist": "Buddhists",
        "target_religion_hindu": "Hindus",
        "target_religion_mormon": "Mormons",
        "target_religion_atheist": "Atheists",
        "target_religion_muslim": "Muslims",
        "target_religion_other": "Other religion",
    },
    "C": {
        "target_origin_specific_country": "A specific country",
        "target_origin_immigrant": "Immigrant",
        "target_origin_migrant_worker": "Migrant worker",
        "target_origin_undocumented": "Undocumented person",
        "target_origin_other": "Other national origin or citizenship status",
    },
    "D": {
        "target_gender_women": "Women",
        "target_gender_men": "Men",
        "target_gender_non_binary": "Non-binary or third gender",
        "target_gender_transgender_women": "Transgender women",
        "target_gender_transgender_men": "Transgender men",
        "target_gender_transgender_unspecified": "Transgender (unspecified)",
        "target_gender_other": "Other gender identity",
    },
    "E": {
        "target_sexuality_bisexual": "Bisexual",
        "target_sexuality_gay": "Gay",
        "target_sexuality_lesbian": "Lesbian",
        "target_sexuality_straight": "Heterosexual",
        "target_sexuality_other": "Other sexual orientation",
    },
    "F": {
        "target_age_children": "Children (0 - 12 years old)",
        "target_age_teenagers": "Adolescents / teenagers (13 - 17)",
        "target_age_young_adults": "Young adults / adults (18 - 39)",
        "target_age_middle_aged": "Middle-aged (40 - 64)",
        "target_age_seniors": "Seniors (65 or older)",
        "target_age_other": "Other age group",
    },
    "G": {
        "target_disability_physical": "People with physical disabilities",
        "target_disability_cognitive": "People with cognitive disorders or learning disabilities",
        "target_disability_neurological": "People with mental health problems",
        "target_disability_visually_impaired": "Visually impaired people",
        "target_disability_hearing_impaired": "Hearing impaired people",
        "target_disability_unspecific": "No specific disability",
        "target_disability_other": "Other disability status",
    },
    "H": {
        "target_politics_republican": "Republican",
        "target_politics_conservative": "Conservative",
        "target_politics_alt_right": "Alt-right",
        "target_politics_democrat": "Democrat",
        "target_politics_liberal": "Liberal",
        "target_politics_green_party": "Green Party",
        "target_politics_socialist": "Socialist",
        "target_politics_communist": "Communist",
        "target_politics_leftist": "Leftist",
        "target_politics_libertarian": "Libertarian",
        "target_politics_other": "Other political ideology",
    },
}


@dataclass(frozen=True)
class MHSAnnotationRecord:
    """Normalized representation of one human or model annotation for one comment."""

    comment_id: int
    judge_id: str
    source_type: str
    text: str
    target_groups: list[str]
    sentiment: str
    respect: str
    insult: str
    humiliate: str
    status: str
    dehumanize: str
    violence: str
    genocide: str
    attack_defend: str
    hate_speech: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate that the record uses the expected source types and item responses."""

        if self.source_type not in {"human", "model"}:
            raise ValueError(f"Unsupported source_type: {self.source_type}")
        if not self.text:
            raise ValueError("text must be non-empty")
        if not self.target_groups:
            raise ValueError("target_groups must be non-empty")
        invalid_target_groups = [group for group in self.target_groups if group not in TARGET_GROUP_CODES]
        if invalid_target_groups:
            raise ValueError(f"target_groups contains invalid codes: {invalid_target_groups}")
        for item_name in ITEM_NAMES:
            value = getattr(self, item_name)
            if value not in ITEM_RESPONSE_LETTERS[item_name]:
                raise ValueError(f"{item_name} has invalid response letter: {value}")


def derive_target_groups(row: pd.Series) -> list[str]:
    """Map truthy target indicators onto the prompt's A-I target-group codes."""

    target_groups = [
        group_code
        for group_code, columns in TARGET_GROUP_COLUMNS.items()
        if any(column_name in row.index and bool(row[column_name]) for column_name in columns)
    ]
    if target_groups:
        return target_groups
    return ["I"]


def normalize_model_annotation(
    comment_id: int,
    judge_id: str,
    text: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> MHSAnnotationRecord:
    """Normalize one model response payload into the shared annotation schema."""

    target_groups = payload.get("target_groups")
    if not isinstance(target_groups, list):
        raise ValueError("target_groups must be a JSON array of uppercase letter codes")

    normalized_target_groups = []
    for group in target_groups:
        if not isinstance(group, str):
            raise ValueError("target_groups entries must be uppercase letter strings")
        normalized_target_groups.append(group.upper())

    record = MHSAnnotationRecord(
        comment_id=int(comment_id),
        judge_id=judge_id,
        source_type="model",
        text=str(text),
        target_groups=normalized_target_groups,
        sentiment=str(payload["sentiment"]).upper(),
        respect=str(payload["respect"]).upper(),
        insult=str(payload["insult"]).upper(),
        humiliate=str(payload["humiliate"]).upper(),
        status=str(payload["status"]).upper(),
        dehumanize=str(payload["dehumanize"]).upper(),
        violence=str(payload["violence"]).upper(),
        genocide=str(payload["genocide"]).upper(),
        attack_defend=str(payload["attack_defend"]).upper(),
        hate_speech=str(payload["hate_speech"]).upper(),
        metadata=metadata,
    )
    record.validate()
    return record


def annotation_record_to_row(
    record: MHSAnnotationRecord,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Convert a validated annotation record into a flat row for storage."""

    row: dict[str, Any] = {
        "comment_id": record.comment_id,
        "judge_id": record.judge_id,
        "provider": str(record.metadata.get("provider", "")),
        "model": str(record.metadata.get("model", "")),
        "text": record.text,
        "target_groups": json.dumps(record.target_groups),
    }
    for item_name in ITEM_NAMES:
        row[item_name] = getattr(record, item_name)
    if include_metadata:
        row["metadata"] = json.dumps(record.metadata, sort_keys=True)
    return row


def _normalize_hf_item_value(item_name: str, value: Any) -> str:
    """Map one Hugging Face numeric item value to the prompt-letter response space."""

    if pd.isna(value):
        raise ValueError(f"{item_name} is missing")
    integer_value = int(value)
    try:
        return HF_VALUE_TO_PROMPT_LETTER[item_name][integer_value]
    except KeyError as exc:
        raise ValueError(f"{item_name} has unsupported HF value: {value}") from exc


def prompt_letter_to_hf_value(item_name: str, value: Any) -> int:
    """Map one prompt-order response letter back into the human numeric coding."""

    normalized_value = str(value).strip().upper()
    try:
        return PROMPT_LETTER_TO_HF_VALUE[item_name][normalized_value]
    except KeyError as exc:
        raise ValueError(f"{item_name} has unsupported prompt letter: {value}") from exc


def normalize_human_annotation(row: pd.Series) -> MHSAnnotationRecord:
    """Convert one raw MHS human-annotation row into the normalized record schema."""

    record = MHSAnnotationRecord(
        comment_id=int(row["comment_id"]),
        judge_id=str(int(row["annotator_id"])),
        source_type="human",
        text=str(row["text"]),
        target_groups=derive_target_groups(row),
        sentiment=_normalize_hf_item_value("sentiment", row["sentiment"]),
        respect=_normalize_hf_item_value("respect", row["respect"]),
        insult=_normalize_hf_item_value("insult", row["insult"]),
        humiliate=_normalize_hf_item_value("humiliate", row["humiliate"]),
        status=_normalize_hf_item_value("status", row["status"]),
        dehumanize=_normalize_hf_item_value("dehumanize", row["dehumanize"]),
        violence=_normalize_hf_item_value("violence", row["violence"]),
        genocide=_normalize_hf_item_value("genocide", row["genocide"]),
        attack_defend=_normalize_hf_item_value("attack_defend", row["attack_defend"]),
        hate_speech=_normalize_hf_item_value("hate_speech", row["hate_speech"]),
        metadata={
            "platform": int(row["platform"]),
            "raw_item_scores": {item_name: int(row[item_name]) for item_name in ITEM_NAMES},
            "target_groups_json": json.dumps(derive_target_groups(row)),
        },
    )
    record.validate()
    return record


def normalize_human_annotations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe of raw MHS human annotations into the repo table format."""

    records = [normalize_human_annotation(row) for _, row in dataframe.iterrows()]
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "comment_id": record.comment_id,
            "judge_id": record.judge_id,
            "source_type": record.source_type,
            "text": record.text,
            "target_groups": json.dumps(record.target_groups),
        }
        for item_name in ITEM_NAMES:
            row[item_name] = record.metadata["raw_item_scores"][item_name]
            row[f"{item_name}_letter"] = getattr(record, item_name)
        row["metadata"] = json.dumps(
            {
                "platform": record.metadata["platform"],
                "target_groups_json": record.metadata["target_groups_json"],
            },
            sort_keys=True,
        )
        rows.append(row)
    return pd.DataFrame(rows)
