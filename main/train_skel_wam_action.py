from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from skel_wam.utils.compatibility import (
    FIXED_TRAINING_SETTINGS,
    validate_fixed_training_settings,
)
from skel_wam.datasets.skel_wam_dataset import SkelWamVideoKeypointQposDataset
from skel_wam.datasets.video_dataset import EpisodeVideoQposDataset
from skel_wam.utils.conditioning import corrupt_condition_tensor
from skel_wam.models.skel_wam import SkelWamActionModel
from skel_wam.models.wan.modules.vae2_2 import Wan2_2_VAE
from skel_wam.utils.flow_matching import FlowMatchingScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the portable Skel-WAM MoT Action head only."
    )
    # Fixed release semantics, also recorded in checkpoint metadata.
    parser.set_defaults(**FIXED_TRAINING_SETTINGS)
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--keypoint_root", default=None)
    parser.add_argument("--action_root", required=True)
    parser.add_argument("--keypoint_stats_path", default=None)
    parser.add_argument("--action_stats_path", required=True)
    parser.add_argument("--vae_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_include", nargs="*", default=None)
    parser.add_argument("--task_exclude", nargs="*", default=None)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--action_video_freq_ratio", type=int, default=4)
    parser.add_argument("--global_sample_stride", type=int, default=1)
    parser.add_argument("--history_keypoint_offsets", type=int, nargs="+", default=[-16, -12, -8, -4])
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 384])
    parser.add_argument("--action_target_frames", type=int, default=33)
    parser.add_argument("--action_dim", type=int, default=50)
    parser.add_argument("--keypoint_dims", type=int, default=3)
    parser.add_argument("--action_condition_mode", choices=["latent_only", "latent_keypoint"], default="latent_keypoint")
    parser.add_argument("--condition_noise_prob", type=float, default=0.5)
    parser.add_argument("--condition_min_timestep", type=float, default=0.0)
    parser.add_argument("--condition_max_timestep", type=float, default=0.2)
    parser.add_argument("--keypoint_condition_noise_prob", type=float, default=0.0)
    parser.add_argument("--keypoint_condition_min_timestep", type=float, default=0.0)
    parser.add_argument("--keypoint_condition_max_timestep", type=float, default=0.0)
    parser.add_argument("--fm_num_train_timesteps", type=int, default=1000)
    parser.add_argument("--action_snr_shift", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--save_step_every", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--lr_scheduler",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--gradient_as_bucket_view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="skel-wam")
    parser.add_argument("--wandb_name", default="skel-wam-action")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["offline", "online", "disabled"],
        default="offline",
    )
    args = parser.parse_args()
    if args.action_condition_mode == "latent_keypoint":
        if not args.keypoint_root:
            parser.error("latent_keypoint requires --keypoint_root")
        if not args.keypoint_stats_path:
            parser.error("latent_keypoint requires --keypoint_stats_path")
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def warmup_cosine_learning_rate(
    schedule_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    effective_warmup_steps = min(warmup_steps, total_steps)
    if effective_warmup_steps > 0 and schedule_step <= effective_warmup_steps:
        return max_lr * schedule_step / effective_warmup_steps
    if total_steps <= effective_warmup_steps:
        return max_lr
    progress = (schedule_step - effective_warmup_steps) / (
        total_steps - effective_warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def checkpoint_state(path: str) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint state in {path}")
    args = payload.get("args", {})
    return state, args if isinstance(args, dict) else {}


@torch.no_grad()
def encode_video(
    vae: Wan2_2_VAE,
    video: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    encoded = vae.encode(
        [item.to(device=device, dtype=dtype) for item in video]
    )
    if encoded is None:
        raise RuntimeError("Wan VAE encode returned None")
    return torch.stack(encoded).to(device=device, dtype=dtype)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> None:
    if not accelerator.is_main_process:
        return
    path.mkdir(parents=True, exist_ok=True)
    state = accelerator.unwrap_model(model).state_dict()
    metadata = vars(args) | {
        "portable_action_checkpoint": True,
        "action_patch_embedding_mode": "portable_frozen",
        "action_target_frames": args.action_target_frames,
    }
    torch.save(
        {"model": state, "step": step, "args": metadata},
        path / f"step_{step:06d}_action_weights.pt",
    )
    torch.save(
        {
            "model": state,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": metadata,
        },
        path / "latest_action_full.pt",
    )
    (path / "trainer_state.json").write_text(
        json.dumps({"step": step, "args": metadata}, indent=2, default=str) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError(
            "min_learning_rate must satisfy "
            "0 <= min_learning_rate <= learning_rate"
        )
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if (args.num_frames - 1) % args.action_video_freq_ratio != 0:
        raise ValueError("num_frames must align with action_video_freq_ratio")
    video_frames = (
        (args.num_frames - 1) // args.action_video_freq_ratio + 1
    )
    if (video_frames - 1) % 4 != 0:
        raise ValueError(
            "Video/keypoint frames must satisfy (frames - 1) % 4 == 0, "
            f"got {video_frames}"
        )
    ddp = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        gradient_as_bucket_view=args.gradient_as_bucket_view,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision={"float32": "no", "float16": "fp16", "bfloat16": "bf16"}[args.dtype],
        kwargs_handlers=[ddp],
    )
    torch.manual_seed(args.seed + accelerator.process_index)
    device = accelerator.device
    dtype = dtype_from_name(args.dtype)

    backbone_state, backbone_args = checkpoint_state(args.backbone_checkpoint)
    backbone_use_keypoint = bool(backbone_args.get("use_keypoint", True))
    requested_use_keypoint = args.action_condition_mode == "latent_keypoint"
    if backbone_use_keypoint != requested_use_keypoint:
        raise ValueError(
            "Action condition mode does not match backbone use_keypoint: "
            f"backbone={backbone_use_keypoint}, "
            f"action_condition_mode={args.action_condition_mode}."
        )
    validate_fixed_training_settings(backbone_args, context="Backbone checkpoint")

    dataset_kwargs = {
        "qpos_root": args.action_root,
        "qpos_target_frames": args.action_target_frames,
        "qpos_stats_path": args.action_stats_path,
        "num_frames": args.num_frames,
        "global_sample_stride": args.global_sample_stride,
        "action_video_freq_ratio": args.action_video_freq_ratio,
        "image_size": tuple(args.image_size),
        "task_include": args.task_include,
        "task_exclude": args.task_exclude,
        "normalize": True,
    }
    if args.action_condition_mode == "latent_keypoint":
        dataset = SkelWamVideoKeypointQposDataset(
            args.dataset_root,
            keypoint_root=args.keypoint_root,
            keypoint_stats_path=args.keypoint_stats_path,
            keypoint_dims=args.keypoint_dims,
            history_keypoint_offsets=args.history_keypoint_offsets,
            **dataset_kwargs,
        )
    else:
        dataset = EpisodeVideoQposDataset(
            args.dataset_root,
            **dataset_kwargs,
        )
    if int(dataset.qpos_dim) != args.action_dim:
        raise ValueError(
            f"Dataset action dim={dataset.qpos_dim}, expected {args.action_dim}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    patch_weight = backbone_state["video_expert.patch_embedding.weight"]
    patch_bias = backbone_state["video_expert.patch_embedding.bias"]
    model = SkelWamActionModel(
        patch_weight=patch_weight.to(device=device, dtype=dtype),
        patch_bias=patch_bias.to(device=device, dtype=dtype),
        action_frames=args.action_target_frames,
        action_dim=args.action_dim,
        keypoint_dims=args.keypoint_dims,
        num_heads=int(backbone_args.get("num_heads", 24)),
        action_condition_mode=args.action_condition_mode,
    )
    del backbone_state, patch_weight, patch_bias
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_args = resume_payload.get("args", {})
        resume_args = resume_args if isinstance(resume_args, dict) else {}
        resume_condition_mode = str(
            resume_args.get("action_condition_mode", "latent_keypoint")
        )
        if resume_condition_mode != args.action_condition_mode:
            raise ValueError(
                "Action resume condition mode mismatch: "
                f"resume={resume_condition_mode}, "
                f"requested={args.action_condition_mode}."
            )
        validate_fixed_training_settings(resume_args, context="Action resume checkpoint")
        model.load_state_dict(resume_payload["model"], strict=True)
        if resume_payload.get("optimizer"):
            optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload.get("step", 0))

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    vae = Wan2_2_VAE(
        vae_pth=str(Path(args.vae_checkpoint_dir) / "Wan2.2_VAE.pth"),
        dtype=dtype,
        device=str(device),
    )
    scheduler = FlowMatchingScheduler(
        num_train_timesteps=args.fm_num_train_timesteps,
        num_inference_steps=20,
        snr_shift=args.action_snr_shift,
    )
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    wandb_run = None
    if accelerator.is_main_process and args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            dir=str(output_dir),
            config=vars(args),
        )
    step = start_step
    progress = tqdm(
        total=args.max_steps,
        initial=step,
        disable=not accelerator.is_local_main_process,
    )
    iterator = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        with accelerator.accumulate(model):
            with torch.no_grad():
                latents = encode_video(
                    vae, batch["video"], device=device, dtype=dtype
                )
                condition_latents, _ = corrupt_condition_tensor(
                    latents,
                    noise_prob=args.condition_noise_prob,
                    min_timestep=args.condition_min_timestep,
                    max_timestep=args.condition_max_timestep,
                    preserve_first_frame_dim=2,
                )
                keypoints = None
                keypoints_valid = None
                if args.action_condition_mode == "latent_keypoint":
                    keypoints = batch["keypoints"].to(
                        device=device, dtype=dtype
                    )
                    keypoints_valid = batch["keypoints_valid"].to(
                        device=device, dtype=torch.bool
                    )
                    keypoints, _ = corrupt_condition_tensor(
                        keypoints,
                        noise_prob=args.keypoint_condition_noise_prob,
                        min_timestep=args.keypoint_condition_min_timestep,
                        max_timestep=args.keypoint_condition_max_timestep,
                        preserve_first_frame_dim=1,
                    )
                    keypoints = torch.where(
                        keypoints_valid.unsqueeze(-1),
                        keypoints,
                        keypoints.new_zeros(()),
                    )
                raw_action = batch["qpos"].to(device=device, dtype=dtype)
                action_valid = torch.isfinite(raw_action).all(dim=-1)
                action = torch.nan_to_num(raw_action)
                flow = scheduler.sample_training_batch(action)
            output = model(
                condition_latents=condition_latents,
                condition_keypoints=keypoints,
                condition_keypoints_valid=keypoints_valid,
                noisy_action=flow.sample,
                action_timestep=flow.timestep,
            )
            error = (
                output["action_velocity"].float()
                - flow.velocity_target.float()
            ).square().mean(dim=-1)
            valid = action_valid.to(error)
            per_sample = (error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            weights = scheduler.training_weight(flow.timestep).to(per_sample)
            loss = (per_sample * weights).mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                if args.lr_scheduler == "warmup_cosine":
                    set_optimizer_learning_rate(
                        optimizer,
                        warmup_cosine_learning_rate(
                            step + 1,
                            total_steps=max(args.max_steps, 1),
                            warmup_steps=args.warmup_steps,
                            max_lr=args.learning_rate,
                            min_lr=args.min_learning_rate,
                        ),
                    )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if accelerator.sync_gradients:
            step += 1
            mean_loss = accelerator.gather(
                loss.detach().float().reshape(1)
            ).mean().item()
            progress.update(1)
            progress.set_postfix(loss=f"{mean_loss:.5f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": mean_loss,
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=step,
                )
            if step % args.save_step_every == 0 or step == args.max_steps:
                accelerator.wait_for_everyone()
                save_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    accelerator=accelerator,
                )
    progress.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
