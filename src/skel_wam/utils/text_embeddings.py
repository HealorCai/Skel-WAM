from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from skel_wam.utils.compatibility import LEGACY_TEXT_EMBEDDINGS_FORMAT


PRECOMPUTED_TEXT_EMBEDDINGS_FORMAT = "skel_wam_prompt_embeddings"
PRECOMPUTED_TEXT_EMBEDDINGS_VERSION = 1


def save_precomputed_text_embeddings(
    path: str | Path,
    embeddings: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, torch.Tensor] = {}
    for prompt, embedding in embeddings.items():
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Precomputed text embedding keys must be non-empty strings")
        if not torch.is_tensor(embedding) or embedding.ndim != 2:
            raise ValueError(
                f"Embedding for prompt {prompt!r} must be a rank-2 tensor"
            )
        normalized[prompt] = embedding.detach().cpu().contiguous()
    torch.save(
        {
            "format": PRECOMPUTED_TEXT_EMBEDDINGS_FORMAT,
            "version": PRECOMPUTED_TEXT_EMBEDDINGS_VERSION,
            "metadata": dict(metadata or {}),
            "embeddings": normalized,
        },
        output_path,
    )


def load_precomputed_text_embeddings(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Precomputed text embedding file does not exist: {input_path}"
        )
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid precomputed embedding payload: {type(payload)}")
    if payload.get("format") not in (
        PRECOMPUTED_TEXT_EMBEDDINGS_FORMAT,
        LEGACY_TEXT_EMBEDDINGS_FORMAT,
    ):
        raise ValueError(
            "Unexpected precomputed embedding format: "
            f"{payload.get('format')!r}"
        )
    if payload.get("version") != PRECOMPUTED_TEXT_EMBEDDINGS_VERSION:
        raise ValueError(
            "Unsupported precomputed embedding version: "
            f"{payload.get('version')!r}"
        )
    raw_embeddings = payload.get("embeddings")
    if not isinstance(raw_embeddings, dict) or not raw_embeddings:
        raise ValueError("Precomputed embedding payload contains no embeddings")
    embeddings: dict[str, torch.Tensor] = {}
    text_dim: int | None = None
    for prompt, embedding in raw_embeddings.items():
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Precomputed embedding keys must be non-empty strings")
        if (
            not torch.is_tensor(embedding)
            or embedding.ndim != 2
            or not torch.is_floating_point(embedding)
        ):
            raise ValueError(
                f"Embedding for prompt {prompt!r} must be a floating rank-2 tensor"
            )
        if not torch.isfinite(embedding).all():
            nonfinite = int((~torch.isfinite(embedding)).sum().item())
            raise ValueError(
                f"Embedding for prompt {prompt!r} contains {nonfinite} "
                "non-finite values"
            )
        if text_dim is None:
            text_dim = int(embedding.shape[-1])
        elif embedding.shape[-1] != text_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions: expected {text_dim}, "
                f"got {embedding.shape[-1]} for prompt {prompt!r}"
            )
        embeddings[prompt] = embedding.detach().cpu().contiguous()
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Precomputed embedding metadata must be a dictionary")
    return embeddings, dict(metadata)


class PrecomputedPromptEmbeddingStore:
    def __init__(
        self,
        embeddings: Mapping[str, torch.Tensor],
        *,
        expected_text_dim: int = 4096,
    ) -> None:
        self.embeddings = dict(embeddings)
        for prompt, embedding in self.embeddings.items():
            if embedding.shape[-1] != expected_text_dim:
                raise ValueError(
                    f"Expected text embedding dim {expected_text_dim}, got "
                    f"{embedding.shape[-1]} for prompt {prompt!r}"
                )

    def encode(
        self,
        prompts: Sequence[str],
        *,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        missing = sorted({prompt for prompt in prompts if prompt not in self.embeddings})
        if missing:
            preview = ", ".join(repr(prompt) for prompt in missing[:3])
            suffix = "" if len(missing) <= 3 else f" and {len(missing) - 3} more"
            raise KeyError(
                "Prompts are missing from the precomputed embedding file: "
                f"{preview}{suffix}"
            )
        return [
            self.embeddings[prompt].to(
                device=output_device,
                dtype=output_dtype,
                non_blocking=True,
            )
            for prompt in prompts
        ]
