from mhs_llms.labels import infer_provider, model_id_to_label, provider_display_name


def test_infer_provider_reads_repo_prefixes() -> None:
    assert infer_provider("openai_gpt-5.4_medium") == "openai"
    assert infer_provider("anthropic_claude-sonnet-4-6_medium") == "anthropic"
    assert infer_provider("xai_grok-4") == "xai"
    assert infer_provider("custom_model") == "unknown"


def test_model_id_to_label_formats_versions_and_reasoning_suffixes() -> None:
    assert model_id_to_label("openai_gpt-5.4-mini_medium") == "GPT-5.4 Mini (Medium)"
    assert model_id_to_label("openai_gpt-4o") == "GPT-4o"
    assert model_id_to_label("anthropic_claude-sonnet-4-6_low") == "Claude Sonnet 4.6 (Low)"
    assert model_id_to_label("anthropic_claude-haiku-4-5") == "Claude Haiku 4.5"


def test_provider_display_name_uses_friendly_names() -> None:
    assert provider_display_name("openai") == "OpenAI"
    assert provider_display_name("anthropic") == "Anthropic"
    assert provider_display_name("unknown") == "Other"
