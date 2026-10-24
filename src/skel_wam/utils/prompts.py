from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping

TASK_LANG_MAP_OPTIMIZED = {
    "Close-Drawer": "Extend the left hand behind the opened wooden door, and push it in an arc motion inward until the door is fully closed and flush against the cabinet structure.",
    "Flip-Mug": "Reach out with the right hand, grab the overturned mug, gently lift it, give it a twist to right it, and then set it back down steadily.",
    "Insert-And-Unload-Cans": "First, pick up the left can and insert it into the left slot, then pick up the right can and insert it into the right slot. Next, reverse the action by unloading the left can from its slot.",
    "Insert-Cans/Left_Right_Deprecated": "Use the left hand to pick up the left can and insert it into the left slot, then use the other hand to pick up the right can and insert it into the right slot.",
    "Insert-Cans": "Use the right hand to pick up the right can and insert it into the right slot, then use the left hand to pick up the left can and insert it into the left slot.",
    "Open-Drawer": "Approach the closed wooden door, firmly grasp its vertical handle, and pull it outward in an arc motion to swing the door open.",
    "Open-Laptop": "Position the right hand at the edge of the laptop screen lid, and lift it upward and backward until the screen is fully open at a viewable angle.",
    "Pour-Balls": "Use the right hand to grasp the cup containing the balls, lift it above the target bowl, and tilt the cup until all the balls pour out completely into the bowl.",
    "Push-Box": "Place the hand against the side of the box and apply steady pressure, sliding the box across the surface until it perfectly aligns with the marker on the table.",
    "Sort-Cans/Single_hand_first_deprecated": "Identify and pick up the Sprite cans to place them into the left box, then pick up the orange soda cans and place them into the right box for sorting.",
    "Sort-Cans": "Sort the cans by picking up the green Sprite cans and placing them into the left box, and picking up the orange cans to place them into the right box.",
    "Stack-Can-Into-Drawer": "Use the right hand to pick up the can and place it carefully onto the saucer in the open cabinet. then use the left hand to push the cabinet door inward until it is fully closed.",
    "Stack-Can": "Use the right hand to pick up the designated can, move it above the saucer, and lower it carefully to stack it right in the center of the saucer.",
    "Unload-Cans": "Reach into the slots to remove the can on the right side first, and then proceed to remove the remaining can from the left side.",
}

DEFAULT_PROMPT_TEMPLATE = "An egocentric video of a robot executing the following task, with a skeletal overlay on the hands: {task}"
DEFAULT_EMPTY_TASK_NAME = "perform the task"


def build_prompts(
    tasks: str | Iterable[str],
    *,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    empty_task_name: str = DEFAULT_EMPTY_TASK_NAME,
    task_lang_map: bool | Mapping[str, str] = True,
) -> list[str]:
    if isinstance(tasks, str):
        tasks = [tasks]
    prompts = []
    for task in tasks:
        task_name = str(task).strip()
        if isinstance(task_lang_map, Mapping):
            task_text = task_lang_map.get(task_name, task_name or empty_task_name)
        elif task_lang_map:
            task_text = TASK_LANG_MAP_OPTIMIZED.get(task_name, task_name or empty_task_name)
        else:
            task_text = task_name or empty_task_name
        prompts.append(template.format(task=task_text))
    return prompts


def load_task_lang_map(path: str | Path) -> dict[str, str]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Task language map does not exist: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Task language map must be a JSON object: {input_path}")
    result: dict[str, str] = {}
    for raw_task, raw_language in payload.items():
        if not isinstance(raw_task, str) or not raw_task.strip():
            raise ValueError(f"Task language map contains an invalid key: {raw_task!r}")
        if not isinstance(raw_language, str) or not raw_language.strip():
            raise ValueError(
                f"Task language for {raw_task!r} must be a non-empty string"
            )
        result[raw_task.strip()] = raw_language.strip()
    if not result:
        raise ValueError(f"Task language map is empty: {input_path}")
    return result


def _stable_token_id(token: str, vocab_size: int) -> int:
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be greater than 1, got {vocab_size}")
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (vocab_size - 1) + 1


def prompt_token_ids(prompts: Iterable[str], *, vocab_size: int, max_tokens: int) -> list[list[int]]:
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    rows = []
    for prompt in prompts:
        words = str(prompt).lower().replace(",", " ").replace(".", " ").replace(":", " ").split()
        ids = [_stable_token_id(word, vocab_size) for word in words[:max_tokens]]
        ids.extend([0] * (max_tokens - len(ids)))
        rows.append(ids)
    return rows
