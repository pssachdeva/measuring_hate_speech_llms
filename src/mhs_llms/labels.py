"""Helpers for turning internal model ids into plot-friendly labels."""


PROVIDER_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "deepseek": "DeepSeek",
    "xai": "xAI",
    "unknown": "Other",
}

_PROVIDER_PREFIX_TO_SLUG = {
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
    "xhigh": "XHigh",
    "opus": "Opus",
    "pro": "Pro",
    "sonnet": "Sonnet",
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

    normalized = model_id.strip()
    provider_slug = infer_provider(normalized)
    provider_prefix = next(
        (prefix for prefix, slug in _PROVIDER_PREFIX_TO_SLUG.items() if slug == provider_slug),
        "",
    )
    if provider_prefix and normalized.startswith(provider_prefix):
        normalized = normalized[len(provider_prefix) :]

    base_name, reasoning_suffix = _split_reasoning_suffix(normalized)
    base_label = _format_base_name(base_name)
    if reasoning_suffix is None:
        return base_label
    return f"{base_label} ({_TOKEN_OVERRIDES.get(reasoning_suffix, reasoning_suffix.title())})"


def _split_reasoning_suffix(model_name: str) -> tuple[str, str | None]:
    """Split off a trailing reasoning suffix when one is present."""

    parts = model_name.split("_")
    if len(parts) < 2:
        return model_name, None

    suffix = parts[-1]
    if suffix not in {"none", "low", "medium", "high", "xhigh"}:
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
