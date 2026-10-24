from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
from torch.utils.data import Dataset

from skel_wam.datasets.general_sample import (
    GeneralMixtureSample,
    GeneralSourceSample,
    ValidationMode,
    validate_general_mixture_sample,
)


@dataclass(frozen=True)
class DatasetSource:
    """A named general-schema dataset and its relative sampling weight."""

    name: str
    dataset: Dataset[GeneralSourceSample]
    weight: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset source name must not be empty")
        if not isinstance(self.weight, (int, float)) or not self.weight > 0:
            raise ValueError(
                f"Dataset source {self.name!r} weight must be positive, "
                f"got {self.weight!r}"
            )
        if len(self.dataset) <= 0:
            raise ValueError(f"Dataset source {self.name!r} must not be empty")


class MixtureIndex(NamedTuple):
    source_id: int
    sample_index: int


class MultiSourceDataset(Dataset[GeneralMixtureSample]):
    """Route sampler-produced ``(source_id, sample_index)`` pairs to datasets."""

    def __init__(
        self,
        sources: list[DatasetSource] | tuple[DatasetSource, ...],
        *,
        validation: ValidationMode = "first",
    ) -> None:
        if not sources:
            raise ValueError("At least one dataset source is required")
        names = [source.name for source in sources]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Dataset source names must be unique: {duplicates}")
        if validation not in ("none", "first", "always"):
            raise ValueError(f"Unknown sample validation mode: {validation!r}")

        self.sources = tuple(sources)
        self.validation = validation
        self._validated_sources: set[int] = set()
        self.source_lengths = tuple(len(source.dataset) for source in self.sources)
        self.source_weights = tuple(float(source.weight) for source in self.sources)
        weights = torch.tensor(self.source_weights, dtype=torch.float64)
        self.source_probabilities = weights / weights.sum()

    def __len__(self) -> int:
        # Sampling is with replacement; this length describes the underlying
        # addressable data, while the sampler controls virtual epoch length.
        return sum(self.source_lengths)

    def __getitem__(
        self,
        index: MixtureIndex | tuple[int, int],
    ) -> GeneralMixtureSample:
        if not isinstance(index, (tuple, list)) or len(index) != 2:
            raise TypeError(
                "MultiSourceDataset indices must be (source_id, sample_index) pairs; "
                "use a mixture sampler instead of integer indices"
            )
        source_id, sample_index = int(index[0]), int(index[1])
        if source_id < 0 or source_id >= len(self.sources):
            raise IndexError(
                f"source_id {source_id} out of range for {len(self.sources)} sources"
            )
        source = self.sources[source_id]
        source_length = self.source_lengths[source_id]
        if sample_index < 0:
            sample_index += source_length
        if sample_index < 0 or sample_index >= source_length:
            raise IndexError(
                f"sample_index {sample_index} out of range for source "
                f"{source.name!r} of size {source_length}"
            )

        raw = source.dataset[sample_index]
        sample = cast(GeneralMixtureSample, dict(raw))
        sample["source_id"] = torch.tensor(source_id, dtype=torch.long)
        sample["source_index"] = torch.tensor(sample_index, dtype=torch.long)
        sample["source_name"] = source.name

        should_validate = self.validation == "always" or (
            self.validation == "first" and source_id not in self._validated_sources
        )
        if should_validate:
            validate_general_mixture_sample(sample, source_name=source.name)
            self._validated_sources.add(source_id)
        return sample
