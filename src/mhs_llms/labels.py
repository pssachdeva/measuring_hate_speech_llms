"""Helpers for turning internal model ids into plot-friendly labels."""


PROVIDER_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "deepseek": "DeepSeek",
    "minimax": "MiniMax",
    "moonshotai": "Moonshot AI",
    "openrouter": "OpenRouter",
    "qwen": "Qwen",
    "xiaomi": "Xiaomi",
    "xai": "xAI",
    "zai": "Z.ai",
    "unknown": "Other",
}

_PROVIDER_PREFIX_TO_SLUG = {
    "openrouter_deepseek_": "deepseek",
    "openrouter_minimax_": "minimax",
    "openrouter_moonshotai_": "moonshotai",
    "openrouter_qwen_": "qwen",
    "openrouter_xiaomi_": "xiaomi",
    "openrouter_z-ai_": "zai",
    "openrouter_": "openrouter",
    "anthropic_": "anthropic",
    "openai_": "openai",
    "google_": "google",
    "deepseek_": "deepseek",
    "xai_": "xai",
}

_TOKEN_OVERRIDES = {
    "claude": "Claude",
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
    "gpt": "GPT",
    "grok": "Grok",
    "haiku": "Haiku",
    "mini": "Mini",
    "nano": "Nano",
    "none": "None",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "minimal": "Minimal",
    "xhigh": "XHigh",
    "opus": "Opus",
    "pro": "Pro",
    "qwen": "Qwen",
    "reasoning": "Reasoning",
    "sonnet": "Sonnet",
}

_MODEL_LABEL_OVERRIDES = {
    "openrouter_deepseek_deepseek-v3.2": "DeepSeek V3.2",
    "openrouter_minimax_minimax-m2.5": "MiniMax M2.5",
    "openrouter_moonshotai_kimi-k2.5": "Kimi K2.5",
    "openrouter_qwen_qwen3.5-122b-a10b": "Qwen3.5 122B A10B",
    "openrouter_xiaomi_mimo-v2-pro": "MiMo V2 Pro",
    "openrouter_z-ai_glm-5-turbo": "GLM-5 Turbo",
}


def infer_provider(model_id: str) -> str:
    """Infer the provider slug from one repository model id."""

    normalized = model_id.strip()
    for prefix, provider_slug in _PROVIDER_PREFIX_TO_SLUG.items():
        if normalized.startswith(prefix):
            return provider_slug
    return "unknown"


def provider_display_name(provider_slug: str) -> str:
    """Return the human-readable provider name for a provider slug."""

    return PROVIDER_DISPLAY_NAMES.get(provider_slug, PROVIDER_DISPLAY_NAMES["unknown"])


def model_id_to_label(model_id: str) -> str:
    """Convert one internal model id into a concise plot label."""

    return _model_id_to_label(model_id=model_id, include_reasoning=True)


def model_id_to_plot_label(model_id: str) -> str:
    """Convert one internal model id into a plot label without reasoning effort."""

    return _model_id_to_label(model_id=model_id, include_reasoning=False)


def _model_id_to_label(model_id: str, include_reasoning: bool) -> str:
    """Convert one model id into a label, optionally including reasoning effort."""

    normalized = model_id.strip()
    if normalized in _MODEL_LABEL_OVERRIDES:
        return _MODEL_LABEL_OVERRIDES[normalized]

    provider_slug = infer_provider(normalized)
    provider_prefix = next(
        (prefix for prefix, slug in _PROVIDER_PREFIX_TO_SLUG.items() if slug == provider_slug),
        "",
    )
    if provider_prefix and normalized.startswith(provider_prefix):
        normalized = normalized[len(provider_prefix) :]

    base_name, reasoning_suffix = _split_reasoning_suffix(normalized)
    base_label = _format_base_name(base_name)
    if reasoning_suffix is None or not include_reasoning:
        return base_label
    return f"{base_label} ({_TOKEN_OVERRIDES.get(reasoning_suffix, reasoning_suffix.title())})"


def _split_reasoning_suffix(model_name: str) -> tuple[str, str | None]:
    """Split off a trailing reasoning suffix when one is present."""

    parts = model_name.split("_")
    if len(parts) < 2:
        return model_name, None

    suffix = parts[-1]
    if suffix not in {"none", "low", "medium", "high", "minimal", "xhigh"}:
        return model_name, None
    return "_".join(parts[:-1]), suffix


def _format_base_name(model_name: str) -> str:
    """Format the provider-stripped portion of a model id."""

    raw_tokens = [token for token in model_name.split("-") if token]
    combined_tokens = _combine_numeric_version_tokens(raw_tokens)
    formatted_tokens = [_format_token(token) for token in combined_tokens]
    if len(formatted_tokens) >= 2 and formatted_tokens[0] == "GPT":
        return f"{formatted_tokens[0]}-{formatted_tokens[1]}" + (
            f" {' '.join(formatted_tokens[2:])}" if len(formatted_tokens) > 2 else ""
        )
    return " ".join(formatted_tokens)


def _combine_numeric_version_tokens(tokens: list[str]) -> list[str]:
    """Merge adjacent numeric version tokens into dotted version strings."""

    combined_tokens: list[str] = []
    current_index = 0
    while current_index < len(tokens):
        current_token = tokens[current_index]
        if _is_numeric_version_token(current_token):
            version_tokens = [current_token]
            look_ahead_index = current_index + 1
            while look_ahead_index < len(tokens) and tokens[look_ahead_index].isdigit():
                version_tokens.append(tokens[look_ahead_index])
                look_ahead_index += 1
            combined_tokens.append(".".join(version_tokens))
            current_index = look_ahead_index
            continue

        combined_tokens.append(current_token)
        current_index += 1
    return combined_tokens


def _is_numeric_version_token(token: str) -> bool:
    """Return whether a token looks like one model version component."""

    return any(character.isdigit() for character in token)


def _format_token(token: str) -> str:
    """Format one token from a provider-stripped model id."""

    if token in _TOKEN_OVERRIDES:
        return _TOKEN_OVERRIDES[token]
    if _is_numeric_version_token(token):
        return token.upper() if token.isalpha() else token
    return token.title()
