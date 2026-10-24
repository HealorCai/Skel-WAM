from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from skel_wam.models.wan.modules.t5 import T5EncoderModel
from skel_wam.utils.prompts import (
    DEFAULT_EMPTY_TASK_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    build_prompts,
)
from skel_wam.utils.text_embeddings import save_precomputed_text_embeddings


DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints/Wan-AI/Wan2.2-TI2V-5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute exact T5 prompt embeddings for Skel-WAM."
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument(
        "--tasks_file",
        default=None,
        help="UTF-8 text file with one task per line, or a JSON list of tasks.",
    )
    parser.add_argument("--prompts", nargs="*", default=[])
    parser.add_argument(
        "--prompts_file",
        default=None,
        help="UTF-8 text file with one exact prompt per line, or a JSON list.",
    )
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--empty_task_name", default=DEFAULT_EMPTY_TASK_NAME)
    parser.add_argument("--disable_task_lang_map", action="store_true")
    parser.add_argument(
        "--checkpoint_dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
    )
    parser.add_argument("--t5_checkpoint_path", default=None)
    parser.add_argument("--t5_tokenizer_path", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_string_list(path: str | Path) -> list[str]:
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Prompt/task list does not exist: {input_path}")
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
        if not isinstance(payload, list) or not all(
            isinstance(item, str) for item in payload
        ):
            raise ValueError(f"Expected a JSON list of strings in {input_path}")
        return payload
    with input_path.open("r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _unique_nonempty(strings: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in strings if value.strip()))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    tasks = list(args.tasks)
    if args.tasks_file:
        tasks.extend(_load_string_list(args.tasks_file))
    exact_prompts = list(args.prompts)
    if args.prompts_file:
        exact_prompts.extend(_load_string_list(args.prompts_file))
    tasks = _unique_nonempty(tasks)
    exact_prompts = _unique_nonempty(exact_prompts)
    generated_prompts = build_prompts(
        tasks,
        template=args.prompt_template,
        empty_task_name=args.empty_task_name,
        task_lang_map=not args.disable_task_lang_map,
    )
    prompts = _unique_nonempty(generated_prompts + exact_prompts)
    if not prompts:
        raise ValueError(
            "No prompts were provided; use --tasks/--tasks_file or "
            "--prompts/--prompts_file."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    checkpoint_path = (
        Path(args.t5_checkpoint_path).expanduser()
        if args.t5_checkpoint_path
        else checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    tokenizer_path = (
        Path(args.t5_tokenizer_path).expanduser()
        if args.t5_tokenizer_path
        else checkpoint_dir / "google/umt5-xxl"
    )
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        shard_fn=None,
    )
    text_encoder.model.eval().requires_grad_(False)

    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            batch_embeddings = text_encoder(batch_prompts, device)
            for prompt, embedding in zip(batch_prompts, batch_embeddings):
                embeddings[prompt] = embedding.detach().cpu().contiguous()
            print(
                f"encoded {min(start + len(batch_prompts), len(prompts))}/"
                f"{len(prompts)} prompts",
                flush=True,
            )

    save_precomputed_text_embeddings(
        args.output_path,
        embeddings,
        metadata={
            "t5_checkpoint_path": str(checkpoint_path),
            "t5_tokenizer_path": str(tokenizer_path),
            "dtype": args.dtype,
            "prompt_template": args.prompt_template,
            "empty_task_name": args.empty_task_name,
            "task_lang_map": not args.disable_task_lang_map,
            "num_prompts": len(prompts),
        },
    )
    print(f"saved {len(prompts)} embeddings to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
