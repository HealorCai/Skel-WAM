from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import Dataset

from skel_wam.datasets.general_sample import (
    GENERAL_SOURCE_SAMPLE_KEYS,
    GeneralSourceSample,
    ValidationMode,
    as_general_source_sample,
)
from skel_wam.utils.prompts import (
    DEFAULT_EMPTY_TASK_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    build_prompts,
    load_task_lang_map,
)


class VideoKeypointSourceAdapter(Dataset[GeneralSourceSample]):
    """Expose EgoVLA video/keypoint data without qpos through one schema."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        empty_task_name: str = DEFAULT_EMPTY_TASK_NAME,
        task_lang_map: bool | Mapping[str, str] = True,
        task_lang_map_path: str | None = None,
        validation: ValidationMode = "first",
    ) -> None:
        self.dataset = dataset
        self.prompt_template = str(prompt_template)
        self.empty_task_name = str(empty_task_name)
        if task_lang_map_path is not None:
            if task_lang_map is not True:
                raise ValueError(
                    "Specify task_lang_map_path without overriding task_lang_map"
                )
            self.task_lang_map: bool | Mapping[str, str] = load_task_lang_map(
                task_lang_map_path
            )
        else:
            self.task_lang_map = task_lang_map
        self.validation = validation
        self._validated = False

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> GeneralSourceSample:
        raw = self.dataset[index]
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"EgoVLA dataset must return a mapping, got {type(raw).__name__}"
            )
        if "task" not in raw:
            raise KeyError("EgoVLA sample is missing required field 'task'")

        adapted = {
            key: raw[key]
            for key in GENERAL_SOURCE_SAMPLE_KEYS
            if key != "language_prompts"
        }
        adapted["language_prompts"] = build_prompts(
            str(raw["task"]),
            template=self.prompt_template,
            empty_task_name=self.empty_task_name,
            task_lang_map=self.task_lang_map,
        )[0]
        sample, self._validated = as_general_source_sample(
            adapted,
            source_name="egovla",
            validation=self.validation,
            already_validated=self._validated,
        )
        return sample


# Compatibility alias for existing imports and source configurations.
EgoVlaGeneralDataset = VideoKeypointSourceAdapter
