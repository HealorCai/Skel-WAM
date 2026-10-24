#!/usr/bin/env python3
"""Precompute T5 embeddings for real robot or robot-human cotrain prompts.

Task names are discovered from processed video roots, mapped through
``configs/real/lang_map.json``, and placed inside the same shared robot prompt
template used by both robot and human cotrain sources.

Example:

    cd Skel-WAM
    python utils/real_utils/precompute_text_embeddings.py
    python utils/real_utils/precompute_text_embeddings.py --mode human_robot

Outputs are saved under ``checkpoints/skel_wam/training_assets``.
Use ``--wo-overlay`` for prompts without a skeletal overlay.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_ROBOT_ROOT = REPO_ROOT / "data/real_robot/video_384x384"
DEFAULT_HUMAN_ROOT = REPO_ROOT / "data/real_human/video_384x384"
DEFAULT_LANG_MAP = REPO_ROOT / "configs/real/lang_map.json"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints/Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_PROMPT_TEMPLATE = (
    "An egocentric video of a robot executing the following task, "
    "with a skeletal overlay on the hands: {task}"
)


def _load_prompt_helpers():
    """Load the lightweight prompt module without importing skel_wam.__init__."""
    prompts_path = SRC_ROOT / "skel_wam/utils/prompts.py"
    spec = importlib.util.spec_from_file_location("skel_wam_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prompt helpers from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_prompts, module.load_task_lang_map


def discover_tasks(dataset_root: Path) -> list[str]:
    tasks = sorted(
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.iterdir()
        if path.is_dir() and any(path.rglob("*.mp4"))
    )
    if not tasks:
        raise FileNotFoundError(f"No task directories with MP4 files under {dataset_root}")
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("robot", "human_robot"), default="robot")
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN_ROOT)
    parser.add_argument("--lang-map", type=Path, default=DEFAULT_LANG_MAP)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--t5-checkpoint-path", type=Path)
    parser.add_argument("--t5-tokenizer-path", type=Path)
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Shared prompt template for both robot and human sources.",
    )
    parser.add_argument("--wo-overlay", action="store_true",
                        help="Use the no-overlay prompt template.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--print-prompts-only",
        action="store_true",
        help="Validate configuration and print exact prompts without loading T5.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.wo_overlay:
        args.prompt_template = "An egocentric video of a robot executing the following task: {task}"
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    robot_root = args.robot_root.expanduser().resolve()
    human_root = args.human_root.expanduser().resolve()
    roots = {"robot": robot_root}
    if args.mode == "human_robot":
        roots["human"] = human_root
    for source_name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{source_name} dataset root does not exist: {root}")
    build_prompts, load_task_lang_map = _load_prompt_helpers()
    lang_map_path = args.lang_map.expanduser().resolve()
    lang_map = load_task_lang_map(lang_map_path)
    tasks_by_source = {name: discover_tasks(root) for name, root in roots.items()}
    all_tasks = sorted({task for tasks in tasks_by_source.values() for task in tasks})
    missing = sorted(set(all_tasks) - set(lang_map))
    if missing:
        raise KeyError(f"Tasks are missing from {lang_map_path}: {missing}")
    templates = {source_name: args.prompt_template for source_name in roots}
    prompts = []
    for source_name, tasks in tasks_by_source.items():
        source_prompts = build_prompts(
            tasks,
            template=templates[source_name],
            task_lang_map=lang_map,
        )
        for task, prompt in zip(tasks, source_prompts):
            print(f"{source_name}/{task}: {prompt}")
        prompts.extend(source_prompts)
    prompts = list(dict.fromkeys(prompts))
    if args.print_prompts_only:
        return

    import torch

    from skel_wam.models.wan.modules.t5 import T5EncoderModel
    from skel_wam.utils.text_embeddings import save_precomputed_text_embeddings

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_path = (
        args.t5_checkpoint_path.expanduser().resolve()
        if args.t5_checkpoint_path
        else checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    tokenizer_path = (
        args.t5_tokenizer_path.expanduser().resolve()
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
    embeddings = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            encoded = text_encoder(batch_prompts, device)
            for prompt, embedding in zip(batch_prompts, encoded):
                embeddings[prompt] = embedding.detach().cpu().contiguous()
            print(f"encoded {min(start + len(batch_prompts), len(prompts))}/{len(prompts)}")

    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else REPO_ROOT
        / "checkpoints/skel_wam/training_assets"
        / (
            "text_embeddings_wo_overlay.pt"
            if args.wo_overlay
            else "text_embeddings.pt"
        )
    )
    save_precomputed_text_embeddings(
        output_path,
        embeddings,
        metadata={
            "mode": args.mode,
            "dataset_roots": {name: str(root) for name, root in roots.items()},
            "tasks_by_source": tasks_by_source,
            "lang_map_path": str(lang_map_path),
            "lang_map": lang_map,
            "prompt_templates": {
                name: templates[name] for name in tasks_by_source
            },
            "t5_checkpoint_path": str(checkpoint_path),
            "t5_tokenizer_path": str(tokenizer_path),
            "dtype": args.dtype,
            "num_prompts": len(prompts),
        },
    )
    print(f"saved {len(prompts)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
