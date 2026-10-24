from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler

from skel_wam.datasets.cotrain_dataset import MixtureIndex


def _validate_mixture_inputs(
    source_lengths: Sequence[int],
    source_weights: Sequence[float],
    num_samples: int,
) -> tuple[tuple[int, ...], torch.Tensor]:
    lengths = tuple(int(length) for length in source_lengths)
    if not lengths:
        raise ValueError("At least one source length is required")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"All source lengths must be positive, got {lengths}")
    if len(lengths) != len(source_weights):
        raise ValueError(
            "source_lengths and source_weights must have the same length, got "
            f"{len(lengths)} and {len(source_weights)}"
        )
    weights = torch.tensor(tuple(float(weight) for weight in source_weights), dtype=torch.float64)
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError(f"All source weights must be finite and positive, got {source_weights}")
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    return lengths, weights / weights.sum()


class WeightedMixtureSampler(Sampler[MixtureIndex]):
    """Sample sources by weight, then sample uniformly within each source."""

    def __init__(
        self,
        source_lengths: Sequence[int],
        source_weights: Sequence[float],
        *,
        num_samples: int,
        seed: int = 0,
    ) -> None:
        self.source_lengths, self.source_probabilities = _validate_mixture_inputs(
            source_lengths, source_weights, num_samples
        )
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self.epoch = int(epoch)

    def _seed_for_epoch(self) -> int:
        return self.seed + self.epoch

    def __iter__(self) -> Iterator[MixtureIndex]:
        generator = torch.Generator()
        generator.manual_seed(self._seed_for_epoch())
        source_ids = torch.multinomial(
            self.source_probabilities,
            num_samples=self.num_samples,
            replacement=True,
            generator=generator,
        )
        for raw_source_id in source_ids.tolist():
            source_id = int(raw_source_id)
            sample_index = int(
                torch.randint(
                    self.source_lengths[source_id],
                    size=(),
                    generator=generator,
                ).item()
            )
            yield MixtureIndex(source_id, sample_index)


class DistributedWeightedMixtureSampler(WeightedMixtureSampler):
    """Rank-local, reproducible weighted sampling for an already-sharded loader.

    Each rank draws ``num_samples_per_rank`` samples with a private generator.
    The DataLoader using this sampler must not be sharded a second time.
    """

    def __init__(
        self,
        source_lengths: Sequence[int],
        source_weights: Sequence[float],
        *,
        num_samples_per_rank: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        if int(num_replicas) <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if int(rank) < 0 or int(rank) >= int(num_replicas):
            raise ValueError(
                f"rank must be in [0, {int(num_replicas)}), got {rank}"
            )
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        super().__init__(
            source_lengths,
            source_weights,
            num_samples=num_samples_per_rank,
            seed=seed,
        )

    def _seed_for_epoch(self) -> int:
        # For fixed world size, every (epoch, rank) pair maps to a distinct seed.
        # This generator is private to the sampler and does not alter model RNG.
        return self.seed + self.epoch * self.num_replicas + self.rank
