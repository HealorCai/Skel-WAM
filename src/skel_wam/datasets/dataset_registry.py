from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from skel_wam.datasets.cotrain_dataset import DatasetSource
from skel_wam.datasets.general_sample import GeneralSourceSample, ValidationMode


DatasetBuilder = Callable[..., Dataset[GeneralSourceSample]]
_DATASET_BUILDERS: dict[str, DatasetBuilder] = {}
_BUILTINS_REGISTERED = False


def register_dataset_builder(name: str) -> Callable[[DatasetBuilder], DatasetBuilder]:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("Dataset builder name must not be empty")

    def decorator(builder: DatasetBuilder) -> DatasetBuilder:
        existing = _DATASET_BUILDERS.get(normalized)
        if existing is not None and existing is not builder:
            raise ValueError(f"Dataset builder {normalized!r} is already registered")
        _DATASET_BUILDERS[normalized] = builder
        return builder

    return decorator


def _register_builtin_dataset_builders() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from skel_wam.datasets.adapters.egovla_general import VideoKeypointSourceAdapter
    from skel_wam.datasets.adapters.real_video_general import (
        VideoSourceAdapter,
    )
    from skel_wam.datasets.skel_wam_dataset import (
        SkelWamVideoKeypointDataset,
    )
    from skel_wam.datasets.video_dataset import EpisodeVideoDataset

    @register_dataset_builder("egovla_general")
    def build_egovla_general(
        *,
        prompt_template: str | None = None,
        empty_task_name: str | None = None,
        task_lang_map: bool = True,
        task_lang_map_path: str | None = None,
        validation: ValidationMode = "first",
        **kwargs: Any,
    ) -> Dataset[GeneralSourceSample]:
        adapter_kwargs: dict[str, object] = {
            "task_lang_map": task_lang_map,
            "validation": validation,
        }
        if task_lang_map_path is not None:
            adapter_kwargs["task_lang_map_path"] = task_lang_map_path
        if prompt_template is not None:
            adapter_kwargs["prompt_template"] = prompt_template
        if empty_task_name is not None:
            adapter_kwargs["empty_task_name"] = empty_task_name
        return VideoKeypointSourceAdapter(
            SkelWamVideoKeypointDataset(**kwargs),
            **adapter_kwargs,
        )

    @register_dataset_builder("real_video_general")
    def build_real_video_general(
        *,
        prompt_template: str | None = None,
        empty_task_name: str | None = None,
        task_lang_map: bool = True,
        task_lang_map_path: str | None = None,
        validation: ValidationMode = "first",
        **kwargs: Any,
    ) -> Dataset[GeneralSourceSample]:
        adapter_kwargs: dict[str, object] = {
            "task_lang_map": task_lang_map,
            "validation": validation,
        }
        if task_lang_map_path is not None:
            adapter_kwargs["task_lang_map_path"] = task_lang_map_path
        if prompt_template is not None:
            adapter_kwargs["prompt_template"] = prompt_template
        if empty_task_name is not None:
            adapter_kwargs["empty_task_name"] = empty_task_name
        return VideoSourceAdapter(
            EpisodeVideoDataset(**kwargs),
            **adapter_kwargs,
        )

    _BUILTINS_REGISTERED = True


def available_dataset_builders() -> tuple[str, ...]:
    _register_builtin_dataset_builders()
    return tuple(sorted(_DATASET_BUILDERS))


def build_registered_dataset(
    dataset_type: str,
    *,
    kwargs: Mapping[str, object] | None = None,
) -> Dataset[GeneralSourceSample]:
    _register_builtin_dataset_builders()
    if dataset_type not in _DATASET_BUILDERS:
        raise KeyError(
            f"Unknown dataset type {dataset_type!r}; "
            f"available={available_dataset_builders()}"
        )
    return _DATASET_BUILDERS[dataset_type](**dict(kwargs or {}))


def build_dataset_sources(
    source_configs: Sequence[Mapping[str, object]],
) -> list[DatasetSource]:
    if not source_configs:
        raise ValueError("At least one source config is required")
    sources: list[DatasetSource] = []
    for index, config in enumerate(source_configs):
        try:
            name = str(config["name"]).strip()
            dataset_type = str(config["type"]).strip()
            weight = float(config["weight"])
        except KeyError as exc:
            raise KeyError(
                f"Source config {index} is missing required field {exc.args[0]!r}"
            ) from exc
        raw_kwargs = config.get("kwargs", {})
        if not isinstance(raw_kwargs, Mapping):
            raise TypeError(
                f"Source config {index} 'kwargs' must be an object, "
                f"got {type(raw_kwargs).__name__}"
            )
        sources.append(
            DatasetSource(
                name=name,
                dataset=build_registered_dataset(dataset_type, kwargs=raw_kwargs),
                weight=weight,
            )
        )
    return sources


def load_dataset_sources_config(
    path: str | Path,
) -> list[DatasetSource]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Cotrain config must contain a JSON object, got {type(payload).__name__}"
        )
    source_configs = payload.get("sources")
    if not isinstance(source_configs, list):
        raise TypeError("Cotrain config field 'sources' must be a list")
    return build_dataset_sources(source_configs)
