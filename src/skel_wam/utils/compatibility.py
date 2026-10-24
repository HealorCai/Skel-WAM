"""Persisted identifiers needed to read existing artifacts.

These strings identify existing files, not public package or
model names. Keep them readable while writing new caches with current names.
"""

LEGACY_TEXT_EMBEDDINGS_FORMAT = "soma_wam_prompt_embeddings"

# Retained in new checkpoint metadata so older readers see the fixed semantics.
FIXED_TRAINING_SETTINGS = {
    "new_valid_logic": True,
    "new_normalize_logic": False,
    "keypoint_condition_dropout_prob": 0.0,
}


def validate_fixed_training_settings(config: dict, *, context: str) -> None:
    """Reject explicit legacy settings that conflict with the release behavior.

    Missing fields are accepted for weight-only artifacts and new readers that
    do not store these historical option names.
    """
    is_cotrain = config.get("dataset_type") == "cotrain"
    if is_cotrain:
        # Historical cotrain datasets took validity from each source, not the
        # top-level parser default. Check embedded source settings when present.
        for source in config.get("cotrain_sources", {}).get("sources", []):
            validate_fixed_training_settings(
                source.get("kwargs", {}),
                context=f"{context}, source {source.get('name', '<unnamed>')}",
            )
    for name, expected in FIXED_TRAINING_SETTINGS.items():
        if name == "new_valid_logic" and is_cotrain and config.get("train_mode", "general") == "general":
            continue
        if name in config and config[name] != expected:
            raise ValueError(
                f"{context}: {name}={config[name]!r} is unsupported; "
                f"this release fixes {name}={expected!r}."
            )
