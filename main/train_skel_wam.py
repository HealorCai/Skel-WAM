from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from skel_wam.utils.compatibility import (
    FIXED_TRAINING_SETTINGS,
    validate_fixed_training_settings,
)
from skel_wam.datasets import (
    DistributedWeightedMixtureSampler,
    MultiSourceDataset,
    build_dataset_sources,
)
from skel_wam.datasets.skel_wam_dataset import (
    SkelWamVideoKeypointDataset,
    SkelWamVideoKeypointQposDataset,
)
from skel_wam.utils.conditioning import corrupt_condition_tensor
from skel_wam.models.skel_wam import (
    SkelWamBackboneModel,
    decode_keypoint_validity,
    load_component_state_dict,
    normalize_keypoint_validity,
)
from skel_wam.models.wan.modules.t5 import T5EncoderModel
from skel_wam.models.wan.modules.vae2_2 import Wan2_2_VAE
from skel_wam.utils.flow_matching import FlowMatchingScheduler
from skel_wam.utils.prompts import (
    DEFAULT_EMPTY_TASK_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    build_prompts,
    load_task_lang_map,
)
from skel_wam.utils.text_embeddings import (
    PrecomputedPromptEmbeddingStore,
    load_precomputed_text_embeddings,
)


DEFAULT_DATASET_ROOT = REPO_ROOT / "data/EgoVLA_SIM/skeleton_video_384x384"
DEFAULT_KEYPOINT_ROOT = REPO_ROOT / "data/EgoVLA_SIM/skeleton_keypoints"
DEFAULT_ACTION_ROOT = REPO_ROOT / "data/EgoVLA_SIM/qpos"
DEFAULT_VAE_CHECKPOINT_DIR = REPO_ROOT / "checkpoints/Wan-AI/Wan2.2-TI2V-5B"


def _process_rss_gib() -> float:
    """Return current process RSS without adding an optional dependency."""
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return float("nan")


def log_startup_timing(
    label: str,
    started_at: float,
    *,
    accelerator: Accelerator,
    synchronize_cuda: bool = True,
) -> float:
    """Print rank-local elapsed time and memory for a startup milestone."""
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    finished_at = time.perf_counter()
    memory = f"rss={_process_rss_gib():.2f}GiB"
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(accelerator.device) / (1024.0**3)
        reserved = torch.cuda.memory_reserved(accelerator.device) / (1024.0**3)
        memory += f" cuda_alloc={allocated:.2f}GiB cuda_reserved={reserved:.2f}GiB"
    print(
        f"[startup-timing][rank {accelerator.process_index}/"
        f"{accelerator.num_processes}] {label}: "
        f"{finished_at - started_at:.3f}s {memory}",
        flush=True,
    )
    return finished_at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Skel-WAM joint latent/keypoint FM model.")
    # Fixed release semantics, also recorded in checkpoint metadata.
    parser.set_defaults(**FIXED_TRAINING_SETTINGS)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--dataset_type",
        default="skel_wam",
        choices=["skel_wam", "cotrain"],
    )
    parser.add_argument(
        "--cotrain_config",
        default=None,
        help="JSON config containing the weighted cotrain source list.",
    )
    parser.add_argument(
        "--cotrain_samples_per_rank",
        type=int,
        default=None,
        help=(
            "Optional rank-local samples per virtual cotrain epoch. By default, "
            "enough samples are generated for the full max_steps run."
        ),
    )
    parser.add_argument(
        "--cotrain_validation",
        default="first",
        choices=["none", "first", "always"],
        help="Runtime validation frequency for the unified cotrain sample schema.",
    )
    parser.add_argument("--keypoint_root", default=str(DEFAULT_KEYPOINT_ROOT))
    parser.add_argument("--action_root", default=str(DEFAULT_ACTION_ROOT))
    parser.add_argument("--action_dataset_key", default="observations/qpos")
    parser.add_argument(
        "--action_exclude_indices",
        type=int,
        nargs="*",
        default=None,
        help="Zero-based raw qpos dimensions excluded from Action training.",
    )
    parser.add_argument("--keypoint_dataset_key", default="keypoints")
    parser.add_argument("--keypoint_stats_path", default="utils/keypoint_stats_new.json")
    parser.add_argument("--action_stats_path", default="utils/qpos_stats.json")
    parser.add_argument("--vae_checkpoint_dir", default=str(DEFAULT_VAE_CHECKPOINT_DIR))
    parser.add_argument("--transformer_checkpoint_dir", default=None)
    parser.add_argument("--t5_checkpoint_path", default=None)
    parser.add_argument("--t5_tokenizer_path", default=None)
    parser.add_argument(
        "--encode_prompts_with_t5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the frozen T5 encoder during training. Disable this to load "
            "all prompt embeddings from --precomputed_text_embeddings_path."
        ),
    )
    parser.add_argument("--precomputed_text_embeddings_path", default=None)
    parser.add_argument(
        "--cache_text_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache frozen T5 prompt embeddings in rank-local CPU memory. "
            "T5 encoding itself always runs on the current rank's GPU."
        ),
    )
    parser.add_argument("--metadata_cache_path", default=None)
    parser.add_argument("--task_include", nargs="*", default=None)
    parser.add_argument("--task_exclude", nargs="*", default=None)

    parser.add_argument("--train_mode", default="general", choices=["general", "robot"])
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--global_sample_stride", type=int, default=1)
    parser.add_argument("--action_video_freq_ratio", type=int, default=4)
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 384])
    parser.add_argument("--history_keypoint_offsets", type=int, nargs="+", default=[-16, -12, -8, -4])
    parser.add_argument("--keypoint_dims", type=int, default=3, choices=[2, 3])
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--empty_task_name", default=DEFAULT_EMPTY_TASK_NAME)
    parser.add_argument("--disable_task_lang_map", action="store_true")
    parser.add_argument(
        "--task_lang_map_path",
        default=None,
        help="Optional JSON task-name to language mapping used before prompt formatting.",
    )
    parser.add_argument("--action_target_frames", type=int, default=49)
    parser.add_argument("--decord_num_threads", type=int, default=1)

    parser.add_argument("--output_dir", default="checkpoints/skel_wam/debug")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help=(
            "Load model weights from --resume but start with a fresh optimizer "
            "and reset the training step to zero."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--keypoint_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for Keypoint Expert parameters.",
    )
    parser.add_argument(
        "--freeze_video_expert",
        action="store_true",
        help=(
            "Freeze the Video Expert and its video-role embedding. The video "
            "branch still participates in the joint forward pass."
        ),
    )
    parser.add_argument(
        "--freeze_keypoint_expert",
        action="store_true",
        help=(
            "Freeze the Keypoint Expert. The keypoint branch still participates "
            "in the joint forward pass."
        ),
    )
    parser.add_argument("--action_lr_multiplier", type=float, default=1.0, help="Learning-rate multiplier for action modules.")
    parser.add_argument("--min_learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--lr_scheduler", default="warmup_cosine", choices=["constant", "warmup_cosine"])
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--vae_dtype",
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Optional VAE encode dtype; defaults to --dtype.",
    )
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--gradient_as_bucket_view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let DDP gradient buckets share storage with parameter gradients.",
    )
    parser.add_argument(
        "--checkpoint_full_layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Checkpoint the complete paired expert layer when gradient "
            "checkpointing is enabled. Disable for mixed-attention-only checkpointing."
        ),
    )
    parser.add_argument("--keypoint_hidden_dim", type=int, default=1024)
    parser.add_argument("--keypoint_ffn_dim", type=int, default=4096)
    parser.add_argument(
        "--keypoint_expert_checkpoint",
        default=None,
        help=(
            "Pretrained Keypoint Expert checkpoint. Required when "
            "--use_keypoint is enabled; omitted for strict w/o-keypoint ablations."
        ),
    )
    parser.add_argument(
        "--keypoint_cond_video_mode",
        choices=["full", "causal"],
        default="full",
    )
    parser.add_argument(
        "--keypoint_use_text_cross_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument(
        "--check_finite",
        action="store_true",
        help="Validate decoded video and VAE latents for NaN/Inf (debugging).",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help=(
            "Run backward with PyTorch autograd anomaly detection. This is "
            "very slow and should only be enabled for a failing smoke run."
        ),
    )
    parser.add_argument("--save_step_every", type=int, default=5000)

    parser.add_argument("--fm_num_train_timesteps", type=int, default=1000, help="Backward-compatible default for both latent/keypoint FM train timesteps.")
    parser.add_argument("--latent_fm_num_train_timesteps", type=int, default=None)
    parser.add_argument("--keypoint_fm_num_train_timesteps", type=int, default=None)
    parser.add_argument("--latent_snr_shift", type=float, default=5.0)
    parser.add_argument("--keypoint_snr_shift", type=float, default=1.0)
    parser.add_argument("--condition_noise_prob", type=float, default=0.5)
    parser.add_argument("--condition_min_timestep", type=float, default=0.0)
    parser.add_argument("--condition_max_timestep", type=float, default=0.5)
    parser.add_argument("--keypoint_condition_noise_prob", type=float, default=0.5)
    parser.add_argument("--keypoint_condition_min_timestep", type=float, default=0.0)
    parser.add_argument("--keypoint_condition_max_timestep", type=float, default=0.5)
    parser.add_argument("--latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--keypoint_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--use_keypoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Keypoint Expert, keypoint FM, and keypoint-to-video "
            "conditioning. Disable for the strict w/o-keypoint ablation."
        ),
    )
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--action_condition_mode",
        default="latent_keypoint",
        choices=["latent_only", "latent_keypoint"],
    )
    parser.add_argument(
        "--action_patch_embedding_mode",
        default="shared",
        choices=["shared", "portable_frozen"],
        help=(
            "Use the Video Expert patch embedding for integrated Action training "
            "(shared), or keep an independent frozen copy for portable Action "
            "checkpoints (portable_frozen)."
        ),
    )
    parser.add_argument("--action_fm_num_train_timesteps", type=int, default=None)
    parser.add_argument("--action_snr_shift", type=float, default=1.0)
    parser.add_argument(
        "--preserve_action_head",
        action="store_true",
        help=(
            "Construct and checkpoint an Action head even when Action loss is "
            "disabled (for example general cotrain resumed from a robot model)."
        ),
    )
    parser.add_argument(
        "--freeze_action_head",
        action="store_true",
        help="Freeze all dedicated Action modules while retaining them in checkpoints.",
    )
    parser.add_argument(
        "--action_dim_override",
        type=int,
        default=None,
        help="Explicit Action dimension when no qpos/action dataset is constructed.",
    )

    parser.add_argument("--wandb_project", default="skel-wam")
    parser.add_argument("--wandb_name", default="skel-wam")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_mode", default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args()
    # Removed legacy shared-backbone/checkpoint switches remain internal false
    # constants so reused data/checkpoint helpers have one unambiguous path.
    args.train_action_only = False
    args.spatical_bias = False
    if args.use_keypoint:
        if not args.keypoint_expert_checkpoint:
            parser.error("--use_keypoint requires --keypoint_expert_checkpoint")
    else:
        if args.action_condition_mode != "latent_only":
            parser.error("--no-use_keypoint requires --action_condition_mode latent_only")
        if args.keypoint_loss_weight != 0.0:
            parser.error("--no-use_keypoint requires --keypoint_loss_weight 0")
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    progress = (schedule_step - effective_warmup_steps) / (total_steps - effective_warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))


def action_module_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    mode = getattr(model, "action_condition_mode", "latent_keypoint")
    prefixes = ["action_patch_embedding.", "action_latent_memory."]
    if mode == "latent_keypoint":
        prefixes.append("action_keypoint_memory.")
    prefixes.append("action_fm_head.")
    return tuple(prefixes)


def apply_expert_freezing(
    model: SkelWamBackboneModel,
    *,
    freeze_video_expert: bool,
    freeze_keypoint_expert: bool,
) -> None:
    if freeze_video_expert:
        model.video_expert.requires_grad_(False)
        model.video_role_embedding.requires_grad_(False)
    if freeze_keypoint_expert and model.keypoint_expert is not None:
        model.keypoint_expert.requires_grad_(False)


def freeze_action_modules(model: SkelWamBackboneModel) -> None:
    prefixes = action_module_prefixes(model)
    matched = 0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(False)
            matched += parameter.numel()
    if matched == 0:
        raise RuntimeError(
            "--freeze_action_head was requested, but the model has no Action parameters"
        )


def load_model_checkpoint(model: torch.nn.Module, state: dict[str, torch.Tensor], *, strict: bool = True) -> None:
    model.load_state_dict(state, strict=strict)


def load_checkpoint_payload(checkpoint_path: str) -> dict[str, object]:
    return torch.load(checkpoint_path, map_location="cpu")


def encode_video_latents(
    vae: Wan2_2_VAE,
    video: torch.Tensor,
    *,
    device: torch.device,
    encode_dtype: torch.dtype,
    output_dtype: torch.dtype,
    check_finite: bool = False,
    sample_labels: list[str] | None = None,
    sample_frame_indices: list[list[int]] | None = None,
    sample_local_starts: list[int] | None = None,
) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(video.shape)}")
    with torch.no_grad():
        video = video.to(device=device, dtype=encode_dtype, non_blocking=True)
        latents = vae.encode([video[i] for i in range(video.shape[0])])
    if latents is None:
        raise RuntimeError("Wan VAE encode returned None")
    stacked = torch.stack(latents).to(device=device)
    if check_finite and not torch.isfinite(stacked).all():
        finite = torch.isfinite(stacked)
        bad_sample_mask = (~finite).flatten(1).any(dim=1)
        bad_sample_indices = bad_sample_mask.nonzero(as_tuple=False).flatten().tolist()
        details = []
        for sample_index in bad_sample_indices:
            latent = stacked[sample_index]
            source_video = video[sample_index]
            label = (
                sample_labels[sample_index]
                if sample_labels is not None and sample_index < len(sample_labels)
                else "unknown"
            )
            details.append(
                {
                    "batch_index": sample_index,
                    "episode_path": label,
                    "local_start_index": (
                        sample_local_starts[sample_index]
                        if sample_local_starts is not None
                        and sample_index < len(sample_local_starts)
                        else None
                    ),
                    "frame_indices": (
                        sample_frame_indices[sample_index]
                        if sample_frame_indices is not None
                        and sample_index < len(sample_frame_indices)
                        else None
                    ),
                    "latent_nonfinite": int((~torch.isfinite(latent)).sum().item()),
                    "latent_numel": latent.numel(),
                    "video_min": float(source_video.min().item()),
                    "video_max": float(source_video.max().item()),
                    "video_mean": float(source_video.float().mean().item()),
                }
            )
        nonfinite = int((~finite).sum().item())
        raise FloatingPointError(
            "Wan VAE encode produced "
            f"{nonfinite}/{stacked.numel()} non-finite latent values; "
            f"bad_samples={details}"
        )
    output = stacked.to(dtype=output_dtype)
    if check_finite and not torch.isfinite(output).all():
        nonfinite = int((~torch.isfinite(output)).sum().item())
        raise FloatingPointError(
            f"Casting VAE latents from {stacked.dtype} to {output_dtype} "
            f"produced {nonfinite}/{output.numel()} non-finite values"
        )
    return output



class T5PromptEmbeddingCache:
    """Rank-local CPU embedding cache backed by a GPU-resident T5 encoder."""

    def __init__(
        self,
        text_encoder: T5EncoderModel,
        *,
        encoder_device: torch.device,
    ) -> None:
        self.text_encoder = text_encoder
        self.encoder_device = encoder_device
        self.cpu_device = torch.device("cpu")
        self.text_encoder.model.to(self.encoder_device)
        self.text_encoder.model.eval().requires_grad_(False)
        self.embeddings: dict[str, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    @torch.no_grad()
    def encode(
        self,
        prompts: list[str],
        *,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        missing_prompts: list[str] = []
        seen_missing: set[str] = set()
        for prompt in prompts:
            if prompt in self.embeddings or prompt in seen_missing:
                self.hits += 1
            else:
                self.misses += 1
                seen_missing.add(prompt)
                missing_prompts.append(prompt)

        if missing_prompts:
            encoded = self.text_encoder(missing_prompts, self.encoder_device)
            if len(encoded) != len(missing_prompts):
                raise RuntimeError(
                    "T5 encoder returned an unexpected number of embeddings: "
                    f"expected {len(missing_prompts)}, got {len(encoded)}"
                )
            for prompt, embedding in zip(missing_prompts, encoded):
                self.embeddings[prompt] = (
                    embedding.detach().to(device=self.cpu_device).contiguous()
                )

        return [
            self.embeddings[prompt].to(
                device=output_device,
                dtype=output_dtype,
                non_blocking=True,
            )
            for prompt in prompts
        ]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)


@torch.no_grad()
def encode_text_prompts(
    text_encoder: T5EncoderModel,
    prompts: list[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Encode a batch on GPU without retaining embeddings between batches."""
    return [
        embedding.to(device=device, dtype=dtype)
        for embedding in text_encoder(prompts, device)
    ]


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> None:
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    model_state = unwrapped.state_dict()
    payload = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }
    weights_payload = {"model": payload["model"], "step": step, "args": vars(args)}
    full_name = "latest_full.pt"
    weights_name = f"step_{step:06d}_weights.pt"
    state_name = "trainer_state.json"
    full_path = output_dir / full_name
    weights_path = output_dir / weights_name
    state_path = output_dir / state_name


    torch.save(payload, full_path)
    torch.save(weights_payload, weights_path)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump({"step": step, "args": vars(args)}, f, ensure_ascii=True, indent=2, default=str)


def validate_args(args: argparse.Namespace) -> None:
    is_cotrain = args.dataset_type == "cotrain"
    if args.resume_weights_only and not args.resume:
        raise ValueError("--resume_weights_only requires --resume.")
    if (
        args.freeze_video_expert
        and args.freeze_keypoint_expert
        and args.train_mode != "robot"
    ):
        raise ValueError(
            "Freezing both experts leaves no trainable modules in general mode."
        )
    if args.keypoint_lr_multiplier <= 0:
        raise ValueError("--keypoint_lr_multiplier must be positive.")
    if args.action_lr_multiplier <= 0:
        raise ValueError("--action_lr_multiplier must be positive.")
    if is_cotrain:
        if args.train_mode != "general":
            raise ValueError("--dataset_type cotrain requires --train_mode general.")
        if not args.cotrain_config:
            raise ValueError("--dataset_type cotrain requires --cotrain_config.")
        if (
            args.cotrain_samples_per_rank is not None
            and args.cotrain_samples_per_rank <= 0
        ):
            raise ValueError("--cotrain_samples_per_rank must be positive.")
        if args.train_action_only:
            raise ValueError("--dataset_type cotrain does not support --train_action_only.")
    if args.preserve_action_head:
        if args.action_dim_override is None or args.action_dim_override <= 0:
            raise ValueError(
                "--preserve_action_head requires a positive --action_dim_override"
            )
    elif args.freeze_action_head:
        raise ValueError("--freeze_action_head requires --preserve_action_head")
    else:
        if args.action_video_freq_ratio <= 0:
            raise ValueError("--action_video_freq_ratio must be positive.")
        if (args.num_frames - 1) % args.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames/action_video_freq_ratio should yield an integer "
                "number of frames including current."
            )
        video_keypoint_frames = (
            (args.num_frames - 1) // args.action_video_freq_ratio + 1
        )
        if (video_keypoint_frames - 1) % 4 != 0:
            raise ValueError(
                "video_keypoint_frames must satisfy "
                "(video_keypoint_frames - 1) % 4 == 0, got "
                f"{video_keypoint_frames}."
            )
    if args.action_target_frames <= 0:
        raise ValueError("--action_target_frames must be positive.")
    if (
        args.train_mode == "robot"
        and args.action_target_frames != args.num_frames
    ):
        raise ValueError(
            "Robot training requires action_target_frames == num_frames, got "
            f"{args.action_target_frames} and {args.num_frames}."
        )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1.")
    if (
        not args.encode_prompts_with_t5
        and not args.precomputed_text_embeddings_path
    ):
        raise ValueError(
            "--no-encode_prompts_with_t5 requires "
            "--precomputed_text_embeddings_path."
        )
    if (
        not args.encode_prompts_with_t5
        and not Path(args.precomputed_text_embeddings_path).expanduser().is_file()
    ):
        raise FileNotFoundError(
            "Precomputed text embedding file does not exist: "
            f"{Path(args.precomputed_text_embeddings_path).expanduser()}"
        )
    if not 0.0 <= args.condition_noise_prob <= 1.0:
        raise ValueError("--condition_noise_prob must be in [0, 1].")
    if not 0.0 <= args.condition_min_timestep <= args.condition_max_timestep <= 1.0:
        raise ValueError("condition timestep range must satisfy 0 <= min <= max <= 1.")
    if not 0.0 <= args.keypoint_condition_noise_prob <= 1.0:
        raise ValueError("--keypoint_condition_noise_prob must be in [0, 1].")
    if not (
        0.0
        <= args.keypoint_condition_min_timestep
        <= args.keypoint_condition_max_timestep
        <= 1.0
    ):
        raise ValueError("keypoint condition timestep range must satisfy 0 <= min <= max <= 1.")
    if args.train_action_only and args.train_mode != "robot":
        raise ValueError("--train_action_only requires --train_mode robot.")




def main() -> None:
    process_started_at = time.perf_counter()
    args = parse_args()
    custom_task_lang_map = (
        load_task_lang_map(args.task_lang_map_path)
        if args.task_lang_map_path
        else None
    )
    if args.disable_task_lang_map and custom_task_lang_map is not None:
        raise ValueError(
            "--disable_task_lang_map and --task_lang_map_path are mutually exclusive"
        )
    validate_args(args)
    seed_everything(args.seed)
    is_cotrain = args.dataset_type == "cotrain"

    mixed_precision = {"float32": "no", "float16": "fp16", "bfloat16": "bf16"}[args.dtype]
    # Action-only freezes the joint backbone and only wraps action trainable params,
    # so unused-parameter detection just adds DDP overhead in that mode.
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        gradient_as_bucket_view=args.gradient_as_bucket_view,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    model_dtype = dtype_from_name(args.dtype)
    vae_dtype = dtype_from_name(args.vae_dtype or args.dtype)
    log_startup_timing(
        "argument parsing + Accelerator init",
        process_started_at,
        accelerator=accelerator,
    )

    cotrain_config_payload = None
    if is_cotrain:
        cotrain_config_path = Path(args.cotrain_config).expanduser().resolve()
        with cotrain_config_path.open("r", encoding="utf-8") as f:
            cotrain_config_payload = json.load(f)
        for source in cotrain_config_payload.get("sources", []):
            source_kwargs = source.get("kwargs", {})
            validate_fixed_training_settings(
                source_kwargs, context=f"Cotrain source {source.get('name', '<unnamed>')}"
            )
            # Accept fixed-value metadata in existing source configurations.
            for name in FIXED_TRAINING_SETTINGS:
                source_kwargs.pop(name, None)

    run_config = dict(vars(args))
    if cotrain_config_payload is not None:
        run_config["cotrain_sources"] = cotrain_config_payload

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=True, indent=2, default=str)
        if cotrain_config_payload is not None:
            with (output_dir / "cotrain_sources.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    cotrain_config_payload,
                    f,
                    ensure_ascii=True,
                    indent=2,
                    default=str,
                )


    wandb_run = None
    stage_started_at = time.perf_counter()
    if accelerator.is_main_process and args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            dir=str(output_dir),
            config=run_config,
        )
    log_startup_timing(
        f"wandb.init (mode={args.wandb_mode})",
        stage_started_at,
        accelerator=accelerator,
        synchronize_cuda=False,
    )

    predict_action = args.train_mode == "robot"
    model_has_action_head = predict_action or args.preserve_action_head
    mixture_sampler = None
    cotrain_source_names: list[str] = []
    cotrain_video_frames: int | None = None
    dataset_started_at = time.perf_counter()
    if is_cotrain:
        stage_started_at = time.perf_counter()
        sources = build_dataset_sources(cotrain_config_payload["sources"])
        log_startup_timing(
            "cotrain source construction",
            stage_started_at,
            accelerator=accelerator,
            synchronize_cuda=False,
        )
        dataset = MultiSourceDataset(
            sources,
            validation=args.cotrain_validation,
        )
        cotrain_source_names = [source.name for source in sources]
        for source in sources:
            stage_started_at = time.perf_counter()
            sample = source.dataset[0]
            log_startup_timing(
                f"cotrain first sample ({source.name})",
                stage_started_at,
                accelerator=accelerator,
                synchronize_cuda=False,
            )
            source_video_frames = int(sample["video"].shape[1])
            if (source_video_frames - 1) % 4 != 0:
                raise ValueError(
                    f"Cotrain source {source.name!r} video frames must satisfy "
                    "(frames - 1) % 4 == 0, got "
                    f"{source_video_frames}."
                )
            if cotrain_video_frames is None:
                cotrain_video_frames = source_video_frames
            elif source_video_frames != cotrain_video_frames:
                raise ValueError(
                    "All cotrain sources must use the same video frame count: "
                    f"expected {cotrain_video_frames}, source "
                    f"{source.name!r} has {source_video_frames}."
                )
            if tuple(sample["video"].shape[-2:]) != tuple(args.image_size):
                raise ValueError(
                    f"Cotrain source {source.name!r} image size "
                    f"{tuple(sample['video'].shape[-2:])} does not match "
                    f"--image_size={tuple(args.image_size)}."
                )
            if not args.use_keypoint:
                unexpected_keypoints = sorted(
                    key for key in sample if "keypoint" in key
                )
                if unexpected_keypoints:
                    raise ValueError(
                        f"Video-only cotrain source {source.name!r} returned "
                        f"keypoint fields: {unexpected_keypoints}"
                    )
                continue
            if "keypoints" not in sample:
                raise ValueError(
                    f"Keypoint-enabled cotrain source {source.name!r} did not "
                    "return keypoint fields."
                )
            if int(sample["keypoints"].shape[0]) != source_video_frames:
                raise ValueError(
                    f"Cotrain source {source.name!r} video/keypoint frame counts "
                    f"differ: {source_video_frames} vs "
                    f"{sample['keypoints'].shape[0]}."
                )
            if sample["keypoints"].shape[-1] != args.keypoint_dims:
                raise ValueError(
                    f"Cotrain source {source.name!r} keypoint dims "
                    f"{sample['keypoints'].shape[-1]} do not match "
                    f"--keypoint_dims={args.keypoint_dims}."
                )

        cotrain_samples_per_rank = args.cotrain_samples_per_rank
        if cotrain_samples_per_rank is None:
            cotrain_samples_per_rank = (
                args.max_steps
                * args.gradient_accumulation_steps
                * args.batch_size
            )
        mixture_sampler = DistributedWeightedMixtureSampler(
            source_lengths=dataset.source_lengths,
            source_weights=dataset.source_weights,
            num_samples_per_rank=cotrain_samples_per_rank,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=mixture_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        base_dataset = None
    else:
        dataset_kwargs = {
            "num_frames": args.num_frames,
            "global_sample_stride": args.global_sample_stride,
            "action_video_freq_ratio": args.action_video_freq_ratio,
            "task_include": args.task_include,
            "task_exclude": args.task_exclude,
            "image_size": tuple(args.image_size),
            "normalize": True,
            "metadata_cache_path": args.metadata_cache_path,
            "decord_num_threads": args.decord_num_threads,
            "keypoint_root": args.keypoint_root,
            "keypoint_dataset_key": args.keypoint_dataset_key,
            "keypoint_stats_path": args.keypoint_stats_path,
            "keypoint_dims": args.keypoint_dims,
            "history_keypoint_offsets": args.history_keypoint_offsets,
        }
        if predict_action:
            # The storage schema is still named qpos; this is the only boundary
            # where the generic Action training API translates to legacy data keys.
            dataset = SkelWamVideoKeypointQposDataset(
                args.dataset_root,
                qpos_root=args.action_root,
                qpos_dataset_key=args.action_dataset_key,
                qpos_target_frames=args.action_target_frames,
                qpos_stats_path=args.action_stats_path,
                qpos_exclude_indices=args.action_exclude_indices,
                **dataset_kwargs,
            )
        else:
            dataset = SkelWamVideoKeypointDataset(args.dataset_root, **dataset_kwargs)
        base_dataset = dataset
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    log_startup_timing(
        "dataset + DataLoader initialization",
        dataset_started_at,
        accelerator=accelerator,
        synchronize_cuda=False,
    )

    stage_started_at = time.perf_counter()
    vae = Wan2_2_VAE(
        vae_pth=str(Path(args.vae_checkpoint_dir) / "Wan2.2_VAE.pth"),
        dtype=vae_dtype,
        device=str(device),
    )
    log_startup_timing(
        "Wan VAE initialization",
        stage_started_at,
        accelerator=accelerator,
    )
    if is_cotrain:
        if cotrain_video_frames is None:
            raise RuntimeError("Cotrain config did not provide any data source")
        video_frames = cotrain_video_frames
    else:
        video_frames = int(base_dataset.video_window_size)
    if (video_frames - 1) % 4 != 0:
        raise ValueError(
            "video_frames must satisfy (video_frames - 1) % 4 == 0, "
            f"got {video_frames}"
        )
    if predict_action and base_dataset is not None:
        action_dim = int(getattr(base_dataset, "qpos_dim", 0))
    elif args.preserve_action_head:
        action_dim = int(args.action_dim_override)
    else:
        action_dim = 0
    transformer_checkpoint_dir = Path(args.transformer_checkpoint_dir or args.vae_checkpoint_dir)
    text_encoder = None
    text_prompt_cache = None
    precomputed_text_store = None
    if not args.train_action_only:
        if args.encode_prompts_with_t5:
            stage_started_at = time.perf_counter()
            text_encoder = T5EncoderModel(
                text_len=512,
                dtype=model_dtype,
                device=device,
                checkpoint_path=str(Path(args.t5_checkpoint_path) if args.t5_checkpoint_path else Path(args.vae_checkpoint_dir) / "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=str(Path(args.t5_tokenizer_path) if args.t5_tokenizer_path else Path(args.vae_checkpoint_dir) / "google/umt5-xxl"),
                shard_fn=None,
            )
            text_encoder.model.eval().requires_grad_(False)
            log_startup_timing(
                "T5 encoder initialization",
                stage_started_at,
                accelerator=accelerator,
            )
            if args.cache_text_embeddings:
                text_prompt_cache = T5PromptEmbeddingCache(
                    text_encoder,
                    encoder_device=device,
                )
        else:
            stage_started_at = time.perf_counter()
            precomputed_embeddings, _ = load_precomputed_text_embeddings(
                args.precomputed_text_embeddings_path
            )
            precomputed_text_store = PrecomputedPromptEmbeddingStore(
                precomputed_embeddings,
                expected_text_dim=4096,
            )
            log_startup_timing(
                "precomputed text embedding load",
                stage_started_at,
                accelerator=accelerator,
            )
    stage_started_at = time.perf_counter()
    model = SkelWamBackboneModel(
        transformer_path=transformer_checkpoint_dir,
        keypoint_frames=video_frames,
        keypoint_dims=args.keypoint_dims,
        keypoint_hidden_dim=args.keypoint_hidden_dim,
        keypoint_ffn_dim=args.keypoint_ffn_dim,
        keypoint_expert_checkpoint=args.keypoint_expert_checkpoint,
        keypoint_cond_video_mode=args.keypoint_cond_video_mode,
        keypoint_use_text_cross_attention=args.keypoint_use_text_cross_attention,
        action_frames=args.action_target_frames,
        action_dim=action_dim,
        predict_action=model_has_action_head,
        torch_dtype=model_dtype,
        device=device,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        checkpoint_full_layer=args.checkpoint_full_layer,
        load_pretrained_transformer=not bool(args.resume),
        action_condition_mode=args.action_condition_mode,
        action_patch_embedding_mode=args.action_patch_embedding_mode,
        use_keypoint=args.use_keypoint,
    )
    log_startup_timing(
        "Skel-WAM model initialization",
        stage_started_at,
        accelerator=accelerator,
    )

    apply_expert_freezing(
        model,
        freeze_video_expert=args.freeze_video_expert,
        freeze_keypoint_expert=args.freeze_keypoint_expert,
    )
    if args.freeze_action_head:
        freeze_action_modules(model)
    if args.train_action_only:
        set_action_only_trainable(model)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters are enabled.")
    action_prefixes = action_module_prefixes(model)
    action_clip_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(action_prefixes)
    ]
    video_keypoint_clip_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith(action_prefixes)
    ]
    optimizer_params: list[torch.nn.Parameter] | list[dict[str, object]]
    if (
        args.keypoint_lr_multiplier == 1.0
        and args.action_lr_multiplier == 1.0
    ):
        # Preserve the historical single-group optimizer layout for old checkpoints.
        optimizer_params = trainable_params
    else:
        video_params = []
        keypoint_params = []
        action_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(action_prefixes):
                action_params.append(param)
            elif name.startswith("keypoint_expert."):
                keypoint_params.append(param)
            else:
                video_params.append(param)
        if (
            args.keypoint_lr_multiplier != 1.0
            and not keypoint_params
            and not args.freeze_keypoint_expert
        ):
            raise RuntimeError(
                "--keypoint_lr_multiplier requires trainable Keypoint Expert parameters."
            )
        if args.action_lr_multiplier != 1.0 and not action_params:
            raise RuntimeError("--action_lr_multiplier requires trainable action modules.")
        optimizer_params = []
        if video_params:
            optimizer_params.append(
                {
                    "params": video_params,
                    "lr": args.learning_rate,
                    "lr_scale": 1.0,
                    "group_name": "video",
                }
            )
        if keypoint_params:
            optimizer_params.append(
                {
                    "params": keypoint_params,
                    "lr": args.learning_rate * args.keypoint_lr_multiplier,
                    "lr_scale": args.keypoint_lr_multiplier,
                    "group_name": "keypoint",
                }
            )
        if action_params:
            optimizer_params.append(
                {
                    "params": action_params,
                    "lr": args.learning_rate * args.action_lr_multiplier,
                    "lr_scale": args.action_lr_multiplier,
                    "group_name": "action",
                }
            )
    stage_started_at = time.perf_counter()
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    log_startup_timing(
        "AdamW initialization",
        stage_started_at,
        accelerator=accelerator,
    )
    start_step = 0
    if args.resume:
        stage_started_at = time.perf_counter()
        payload = load_checkpoint_payload(args.resume)
        log_startup_timing(
            "resume checkpoint torch.load",
            stage_started_at,
            accelerator=accelerator,
            synchronize_cuda=False,
        )
        stage_started_at = time.perf_counter()
        load_component_state_dict(model, payload["model"])
        log_startup_timing(
            "resume model load_state_dict",
            stage_started_at,
            accelerator=accelerator,
        )
        checkpoint_args = payload.get("args")
        checkpoint_args = checkpoint_args if isinstance(checkpoint_args, dict) else {}
        validate_fixed_training_settings(checkpoint_args, context="Resume checkpoint")
        if not args.resume_weights_only and payload.get("optimizer") is not None:
            checkpoint_freezing = (
                bool(checkpoint_args.get("freeze_video_expert", False)),
                bool(checkpoint_args.get("freeze_keypoint_expert", False)),
            )
            requested_freezing = (
                args.freeze_video_expert,
                args.freeze_keypoint_expert,
            )
            if checkpoint_freezing != requested_freezing:
                raise ValueError(
                    "Checkpoint optimizer freezing configuration does not match "
                    f"the current run: checkpoint={checkpoint_freezing}, "
                    f"current={requested_freezing}. Use --resume_weights_only "
                    "to load the model without restoring optimizer state."
                )
            try:
                stage_started_at = time.perf_counter()
                optimizer.load_state_dict(payload["optimizer"])
                log_startup_timing(
                    "resume optimizer load_state_dict",
                    stage_started_at,
                    accelerator=accelerator,
                )
            except (ValueError, RuntimeError) as exc:
                raise ValueError(
                    "Checkpoint optimizer state is incompatible with the current "
                    "trainable parameter groups. Use --resume_weights_only to "
                    "load only the model weights."
                ) from exc
            desired_lr_scales = {
                "shared": 1.0,
                "video": 1.0,
                "keypoint": args.keypoint_lr_multiplier,
                "action": args.action_lr_multiplier,
            }
            for group in optimizer.param_groups:
                group_name = str(group.get("group_name", "shared"))
                if group_name in desired_lr_scales:
                    group["lr_scale"] = desired_lr_scales[group_name]
            if args.lr_scheduler == "constant":
                set_optimizer_learning_rate(optimizer, args.learning_rate)
        if not args.resume_weights_only:
            start_step = int(payload.get("step") or 0)

    stage_started_at = time.perf_counter()
    if is_cotrain:
        # The cotrain sampler is already rank-aware. Preparing this DataLoader
        # would shard it a second time across the global process group.
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    log_startup_timing(
        "accelerator.prepare (DDP)",
        stage_started_at,
        accelerator=accelerator,
    )
    latent_fm_scheduler = FlowMatchingScheduler(
        num_train_timesteps=(
            args.latent_fm_num_train_timesteps
            if args.latent_fm_num_train_timesteps is not None
            else args.fm_num_train_timesteps
        ),
        num_inference_steps=20,
        snr_shift=args.latent_snr_shift,
    )
    keypoint_fm_scheduler = FlowMatchingScheduler(
        num_train_timesteps=(
            args.keypoint_fm_num_train_timesteps
            if args.keypoint_fm_num_train_timesteps is not None
            else args.fm_num_train_timesteps
        ),
        num_inference_steps=20,
        snr_shift=args.keypoint_snr_shift,
    )
    action_fm_scheduler = None
    if predict_action:
        action_fm_scheduler = FlowMatchingScheduler(
            num_train_timesteps=(
                args.action_fm_num_train_timesteps
                if args.action_fm_num_train_timesteps is not None
                else args.fm_num_train_timesteps
            ),
            num_inference_steps=20,
            snr_shift=args.action_snr_shift,
        )

    if accelerator.is_main_process:
        ready_msg = (
            f"Skel-WAM dataset ready: samples={len(dataset)} "
            f"type={args.dataset_type} mode={args.train_mode} "
            f"video/keypoint frames={video_frames} "
            f"batch_size={args.batch_size} grad_accum={args.gradient_accumulation_steps} "
            f"latent_fm(snr_shift={latent_fm_scheduler.snr_shift}, steps={latent_fm_scheduler.num_train_timesteps}) "
            f"keypoint_fm(snr_shift={keypoint_fm_scheduler.snr_shift}, steps={keypoint_fm_scheduler.num_train_timesteps})"
        )
        if is_cotrain:
            source_mix = ", ".join(
                f"{name}={float(dataset.source_probabilities[index]):.4f}"
                for index, name in enumerate(cotrain_source_names)
            )
            ready_msg += (
                f" cotrain_sources=[{source_mix}] "
                f"samples_per_rank={len(mixture_sampler)}"
            )
        if predict_action:
            ready_msg += (
                f" action_dim={action_dim} "
                f"action_fm(snr_shift={action_fm_scheduler.snr_shift}, steps={action_fm_scheduler.num_train_timesteps})"
            )
        tqdm.write(ready_msg)

    step = start_step
    # A resumed cotrain run starts a fresh deterministic sampler stream keyed by
    # its checkpoint step instead of replaying the stream from step zero.
    data_epoch = start_step if mixture_sampler is not None else 0
    if mixture_sampler is not None:
        mixture_sampler.set_epoch(data_epoch)
    first_batch_started_at = time.perf_counter()
    first_batch_pending = True
    first_step_started_at = first_batch_started_at
    first_step_pending = True
    data_iter = iter(loader)
    cotrain_source_counts = (
        torch.zeros(len(cotrain_source_names), device=device, dtype=torch.float64)
        if is_cotrain
        else None
    )
    progress = (
        tqdm(total=args.max_steps, initial=start_step, desc="train_skel_wam", dynamic_ncols=True)
        if accelerator.is_main_process
        else None
    )
    last_time = time.perf_counter()
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            if mixture_sampler is not None:
                data_epoch += 1
                mixture_sampler.set_epoch(data_epoch)
            data_iter = iter(loader)
            batch = next(data_iter)
        if first_batch_pending:
            log_startup_timing(
                "first DataLoader batch",
                first_batch_started_at,
                accelerator=accelerator,
                synchronize_cuda=False,
            )
            first_batch_pending = False

        if predict_action:
            if "qpos" not in batch:
                raise KeyError("Robot dataset must provide the storage field 'qpos'")
            batch["action"] = batch["qpos"]
            if "qpos_is_pad" in batch:
                batch["action_is_pad"] = batch["qpos_is_pad"]

        if cotrain_source_counts is not None:
            source_ids = batch["source_id"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            cotrain_source_counts += torch.bincount(
                source_ids,
                minlength=len(cotrain_source_names),
            ).to(dtype=torch.float64)

        with accelerator.accumulate(model):
            video_batch = batch["video"]
            episode_paths = list(batch.get("episode_path", []))
            sample_frame_indices = (
                batch["frame_indices"].tolist()
                if "frame_indices" in batch
                else None
            )
            sample_local_starts = (
                batch["local_start_index"].tolist()
                if "local_start_index" in batch
                else None
            )
            if args.check_finite and not torch.isfinite(video_batch).all():
                nonfinite = int((~torch.isfinite(video_batch)).sum().item())
                raise FloatingPointError(
                    f"Decoded video batch contains {nonfinite}/"
                    f"{video_batch.numel()} non-finite values; "
                    f"episode_paths={episode_paths}"
                )
            try:
                latents = encode_video_latents(
                    vae,
                    video_batch,
                    device=device,
                    encode_dtype=vae_dtype,
                    output_dtype=model_dtype,
                    check_finite=args.check_finite,
                    sample_labels=episode_paths,
                    sample_frame_indices=sample_frame_indices,
                    sample_local_starts=sample_local_starts,
                )
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"{exc}; episode_paths={episode_paths}"
                ) from exc
            keypoints = None
            kp_valid = None
            history_keypoints = None
            history_keypoints_valid = None
            neutral_keypoint_value = None
            if args.use_keypoint:
                keypoints_raw = batch["keypoints"].to(device=device, dtype=model_dtype, non_blocking=True)
                keypoints_raw = keypoints_raw[..., : args.keypoint_dims]
                if "keypoints_valid" in batch:
                    kp_valid = batch["keypoints_valid"].to(device=device, dtype=torch.bool, non_blocking=True)
                else:
                    kp_valid = torch.isfinite(keypoints_raw).all(dim=-1)
                keypoints = torch.nan_to_num(keypoints_raw, nan=0.0, posinf=0.0, neginf=0.0)
                neutral_keypoint_value = keypoints.new_zeros(())
                history_keypoints = batch["history_keypoints"].to(device=device, dtype=model_dtype, non_blocking=True)
                history_keypoints = history_keypoints[..., : args.keypoint_dims]
                if "history_keypoints_valid" in batch:
                    history_keypoints_valid = batch["history_keypoints_valid"].to(device=device, dtype=torch.bool, non_blocking=True)
                else:
                    history_keypoints_valid = torch.isfinite(history_keypoints).all(dim=-1)
                history_keypoints = torch.nan_to_num(history_keypoints, nan=0.0, posinf=0.0, neginf=0.0)
            text_condition = None
            latent_flow = None
            keypoint_flow = None
            condition_latents = None
            condition_latent_timestep = latents.new_zeros((latents.shape[0],), dtype=torch.float32)
            kp_valid_f = (
                kp_valid.unsqueeze(-1).to(keypoints)
                if args.use_keypoint
                else None
            )
            keypoint_mean = None
            keypoint_std = None
            if args.spatical_bias and "keypoint_mean" in batch and "keypoint_std" in batch:
                keypoint_mean = batch["keypoint_mean"].to(
                    device=device, dtype=model_dtype, non_blocking=True
                )
                keypoint_std = batch["keypoint_std"].to(
                    device=device, dtype=model_dtype, non_blocking=True
                )

            if not args.train_action_only:
                if "language_prompts" in batch:
                    prompts = list(batch["language_prompts"])
                else:
                    prompts = build_prompts(
                        batch["task"],
                        template=args.prompt_template,
                        empty_task_name=args.empty_task_name,
                        task_lang_map=(
                            custom_task_lang_map
                            if custom_task_lang_map is not None
                            else not args.disable_task_lang_map
                        ),
                    )
                if precomputed_text_store is not None:
                    text_condition = precomputed_text_store.encode(
                        prompts,
                        output_device=device,
                        output_dtype=model_dtype,
                    )
                elif text_prompt_cache is not None:
                    text_condition = text_prompt_cache.encode(
                        prompts,
                        output_device=device,
                        output_dtype=model_dtype,
                    )
                else:
                    if text_encoder is None:
                        raise RuntimeError(
                            "Neither T5 nor precomputed text embeddings are available"
                        )
                    text_condition = encode_text_prompts(
                        text_encoder,
                        prompts,
                        device=device,
                        dtype=model_dtype,
                    )

                latent_flow = latent_fm_scheduler.sample_training_batch(latents)
                current_keypoints = None
                current_keypoints_valid = None
                if args.use_keypoint:
                    current_keypoints = keypoints[:, 0:1]
                    current_keypoints_valid = kp_valid[:, 0:1]
                    keypoint_validity_target = normalize_keypoint_validity(
                        kp_valid,
                        dtype=keypoints.dtype,
                    )
                    keypoint_flow_target = torch.cat(
                        [keypoints, keypoint_validity_target.unsqueeze(-1)],
                        dim=-1,
                    )
                    keypoint_flow = keypoint_fm_scheduler.sample_training_batch(
                        keypoint_flow_target
                    )
                    kp_valid_f = kp_valid.unsqueeze(-1).to(keypoint_flow.sample)
                condition_latents, condition_latent_timestep = corrupt_condition_tensor(
                    latents,
                    noise_prob=args.condition_noise_prob,
                    min_timestep=args.condition_min_timestep,
                    max_timestep=args.condition_max_timestep,
                    preserve_first_frame_dim=2,
                )
            action_outputs = None
            action = None
            action_valid = None
            action_flow = None
            if predict_action:
                action_condition_latent_timestep = latents.new_zeros((latents.shape[0],), dtype=torch.float32)
                action_condition_keypoint_timestep = latents.new_zeros((latents.shape[0],), dtype=torch.float32)
                action_raw = batch["action"].to(device=device, dtype=model_dtype, non_blocking=True)
                action_finite = torch.isfinite(action_raw).all(dim=-1)
                action_valid = action_finite
                action = torch.nan_to_num(action_raw, nan=0.0, posinf=0.0, neginf=0.0)
                if action_fm_scheduler is None:
                    raise RuntimeError("action_fm_scheduler is required for action FM training")
                action_flow = action_fm_scheduler.sample_training_batch(action)
                action_condition_latents, action_condition_latent_timestep = corrupt_condition_tensor(
                    latents,
                    noise_prob=args.condition_noise_prob,
                    min_timestep=args.condition_min_timestep,
                    max_timestep=args.condition_max_timestep,
                    preserve_first_frame_dim=2,
                )
                action_condition_keypoints = None
                action_condition_keypoints_valid = None
                if args.action_condition_mode != "latent_only":
                    action_condition_keypoints, action_condition_keypoint_timestep = corrupt_condition_tensor(
                        keypoints,
                        noise_prob=args.keypoint_condition_noise_prob,
                        min_timestep=args.keypoint_condition_min_timestep,
                        max_timestep=args.keypoint_condition_max_timestep,
                        preserve_first_frame_dim=1,
                    )
                    action_condition_keypoints = torch.where(
                        kp_valid_f.bool(), action_condition_keypoints, neutral_keypoint_value
                    )
                    action_condition_keypoints_valid = kp_valid
            if predict_action:
                joint_outputs = None
            if predict_action and not args.train_action_only and args.use_keypoint:
                robot_kwargs = {
                    "noisy_action": action_flow.sample,
                    "action_timestep": action_flow.timestep,
                }
                joint_outputs = model(
                    method="forward_robot",
                    noisy_latents=latent_flow.sample,
                    condition_latents=condition_latents,
                    noisy_keypoints=keypoint_flow.sample[..., : args.keypoint_dims],
                    noisy_keypoint_validity=keypoint_flow.sample[..., args.keypoint_dims],
                    current_keypoints=current_keypoints,
                    history_keypoints=history_keypoints,
                    latent_timestep=latent_flow.timestep,
                    keypoint_timestep=keypoint_flow.timestep,
                    text_emb=text_condition,
                    current_keypoints_valid=current_keypoints_valid,
                    history_keypoints_valid=history_keypoints_valid,
                    action_condition_latents=action_condition_latents,
                    action_condition_keypoints=action_condition_keypoints,
                    action_condition_keypoints_valid=action_condition_keypoints_valid,
                    keypoint_mean=keypoint_mean,
                    keypoint_std=keypoint_std,
                    image_size=tuple(args.image_size),
                    **robot_kwargs,
                )
                action_outputs = joint_outputs
            elif not predict_action and args.use_keypoint:
                joint_outputs = model(
                    method="forward_joint",
                    noisy_latents=latent_flow.sample,
                    condition_latents=condition_latents,
                    noisy_keypoints=keypoint_flow.sample[..., : args.keypoint_dims],
                    noisy_keypoint_validity=keypoint_flow.sample[..., args.keypoint_dims],
                    current_keypoints=current_keypoints,
                    history_keypoints=history_keypoints,
                    latent_timestep=latent_flow.timestep,
                    keypoint_timestep=keypoint_flow.timestep,
                    text_emb=text_condition,
                    current_keypoints_valid=current_keypoints_valid,
                    history_keypoints_valid=history_keypoints_valid,
                    keypoint_mean=keypoint_mean,
                    keypoint_std=keypoint_std,
                    image_size=tuple(args.image_size),
                )
            elif not args.train_action_only:
                joint_outputs = model(
                    method="forward_video_only",
                    noisy_latents=latent_flow.sample,
                    condition_latents=condition_latents,
                    latent_timestep=latent_flow.timestep,
                    text_emb=text_condition,
                )
            loss_latent = latents.new_zeros((), dtype=torch.float32)
            loss_kp = latents.new_zeros((), dtype=torch.float32)
            loss_kp_xy = latents.new_zeros((), dtype=torch.float32)
            loss_kp_z = latents.new_zeros((), dtype=torch.float32)
            loss_kp_validity = latents.new_zeros((), dtype=torch.float32)
            loss_latent_unweighted = latents.new_zeros((), dtype=torch.float32)
            kp_valid_counts = latents.new_zeros((6,), dtype=torch.float32)
            loss = latents.new_zeros((), dtype=torch.float32)
            if not args.train_action_only:
                video_outputs = joint_outputs
                keypoint_outputs = joint_outputs

                latent_velocity_error = (
                    video_outputs["latent_velocity"][:, :, 1:].float()
                    - latent_flow.velocity_target[:, :, 1:].float()
                ).pow(2)
                latent_weight = latent_fm_scheduler.training_weight(
                    latent_flow.timestep
                ).to(latent_velocity_error)
                latent_loss_unweighted_per_sample = (
                    latent_velocity_error.flatten(1).mean(dim=1)
                )
                loss_latent_unweighted = (
                    latent_loss_unweighted_per_sample * latent_weight
                ).mean()
                latent_loss_per_sample = latent_loss_unweighted_per_sample
                loss_latent = (latent_loss_per_sample * latent_weight).mean()

                loss = args.latent_loss_weight * loss_latent + args.keypoint_loss_weight * loss_kp
                if args.use_keypoint:
                    joint_velocity_error = (
                        keypoint_outputs["keypoint_flow_velocity"][:, 1:].float()
                        - keypoint_flow.velocity_target[:, 1:].float()
                    ).pow(2)
                    xyz_velocity_error = joint_velocity_error[..., : args.keypoint_dims]
                    validity_velocity_error = joint_velocity_error[..., args.keypoint_dims]
                    xyz_loss_mask = kp_valid[:, 1:].unsqueeze(-1).to(xyz_velocity_error)
                    loss_kp_per_sample = (
                        (xyz_velocity_error * xyz_loss_mask).flatten(1).sum(dim=1)
                        + validity_velocity_error.flatten(1).sum(dim=1)
                    )
                    xyz_count = (
                        kp_valid[:, 1:].to(xyz_velocity_error).flatten(1).sum(dim=1)
                        * args.keypoint_dims
                    )
                    validity_count = validity_velocity_error.new_full(
                        (validity_velocity_error.shape[0],),
                        validity_velocity_error[0].numel(),
                    )
                    kp_denom = (xyz_count + validity_count).clamp_min(1.0)
                    keypoint_weight = keypoint_fm_scheduler.training_weight(
                        keypoint_flow.timestep
                    ).to(loss_kp_per_sample)
                    loss_kp = ((loss_kp_per_sample / kp_denom) * keypoint_weight).mean()
                    valid_joint_count = kp_valid[:, 1:].flatten(1).sum(dim=1)
                    xy_denom = (
                        valid_joint_count * min(args.keypoint_dims, 2)
                    ).clamp_min(1.0)
                    xy_per_sample = (
                        xyz_velocity_error[..., :2] * xyz_loss_mask
                    ).flatten(1).sum(dim=1) / xy_denom
                    loss_kp_xy = (xy_per_sample * keypoint_weight).mean()
                    if args.keypoint_dims == 3:
                        z_per_sample = (
                            xyz_velocity_error[..., 2:3] * xyz_loss_mask
                        ).flatten(1).sum(dim=1) / valid_joint_count.clamp_min(1.0)
                        loss_kp_z = (z_per_sample * keypoint_weight).mean()
                    validity_per_sample = validity_velocity_error.flatten(1).mean(dim=1)
                    loss_kp_validity = (
                        validity_per_sample * keypoint_weight
                    ).mean()
                    loss = loss + args.keypoint_loss_weight * loss_kp
                    clean_validity_estimate = keypoint_fm_scheduler.estimate_x1(
                        keypoint_flow.sample[..., args.keypoint_dims],
                        keypoint_outputs["keypoint_validity_velocity"],
                        keypoint_flow.timestep,
                    )
                    valid_pred = decode_keypoint_validity(
                        clean_validity_estimate[:, 1:].detach(),
                    )
                    valid_target_bool = kp_valid[:, 1:]
                    kp_valid_counts = torch.stack(
                        [
                            (valid_pred & valid_target_bool).sum(),
                            (valid_pred & ~valid_target_bool).sum(),
                            (~valid_pred & valid_target_bool).sum(),
                            valid_target_bool.sum(),
                            valid_pred.sum(),
                            valid_target_bool.new_ones(valid_target_bool.shape).sum(),
                        ]
                    ).to(dtype=torch.float32)


            loss_action = latents.new_zeros((), dtype=torch.float32)
            if predict_action:
                if action_outputs is None:
                    action_outputs = model(
                        method="forward_action_fm",
                        condition_latents=action_condition_latents,
                        condition_keypoints=action_condition_keypoints,
                        history_keypoints=history_keypoints,
                        noisy_action=action_flow.sample,
                        action_timestep=action_flow.timestep,
                        text_emb=text_condition,
                        condition_keypoints_valid=action_condition_keypoints_valid,
                        history_keypoints_valid=history_keypoints_valid,
                    )
                loss_action_per = (
                    action_outputs["action_velocity"].float() - action_flow.velocity_target.float()
                ).pow(2).mean(dim=-1)
                action_valid_f = action_valid.to(loss_action_per)
                loss_action_per_sample = (loss_action_per * action_valid_f).sum(dim=1)
                action_denom = action_valid_f.sum(dim=1).clamp_min(1.0)
                if action_fm_scheduler is None:
                    raise RuntimeError("action_fm_scheduler is required for action FM training")
                action_weight = action_fm_scheduler.training_weight(action_flow.timestep).to(loss_action_per_sample)
                loss_action = ((loss_action_per_sample / action_denom) * action_weight).mean()
                loss = loss + args.action_loss_weight * loss_action

            if args.detect_anomaly:
                with torch.autograd.detect_anomaly(check_nan=True):
                    accelerator.backward(loss)
            else:
                accelerator.backward(loss)
            if accelerator.sync_gradients:
                # Clip the two gradient domains independently.  Otherwise a
                # large Action gradient would rescale Video/Keypoint gradients
                # (or vice versa) even though their computation graphs are
                # isolated at the Action patch features.
                video_keypoint_grad_norm = None
                action_grad_norm = None
                if video_keypoint_clip_params:
                    video_keypoint_grad_norm = accelerator.clip_grad_norm_(
                        video_keypoint_clip_params,
                        args.max_grad_norm,
                    )
                if action_clip_params:
                    action_grad_norm = accelerator.clip_grad_norm_(
                        action_clip_params,
                        args.max_grad_norm,
                    )
                nonfinite_grad_domains = []
                if (
                    video_keypoint_grad_norm is not None
                    and not torch.isfinite(video_keypoint_grad_norm)
                ):
                    nonfinite_grad_domains.append(
                        "video_keypoint="
                        f"{video_keypoint_grad_norm.detach().float().item()}"
                    )
                if (
                    action_grad_norm is not None
                    and not torch.isfinite(action_grad_norm)
                ):
                    nonfinite_grad_domains.append(
                        f"action={action_grad_norm.detach().float().item()}"
                    )
                if nonfinite_grad_domains:
                    # Never let AdamW turn a transient non-finite gradient into
                    # permanently corrupted model weights.
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Refusing optimizer.step() because gradient norm is "
                        f"non-finite at training_step={step + 1}: "
                        + ", ".join(nonfinite_grad_domains)
                        + "; losses="
                        + str(
                            {
                                "total": float(loss.detach().float().item()),
                                "latent": float(loss_latent.detach().float().item()),
                                "keypoint": float(loss_kp.detach().float().item()),
                                "action": float(loss_action.detach().float().item()),
                            }
                        )
                    )
                if args.lr_scheduler == "warmup_cosine":
                    schedule_step = step + 1
                    # schedule_step = step - start_step + 1
                    set_optimizer_learning_rate(
                        optimizer,
                        warmup_cosine_learning_rate(
                            schedule_step,
                            total_steps=max(args.max_steps, 1),
                            # total_steps=max(args.max_steps - start_step, 1),
                            warmup_steps=args.warmup_steps,
                            max_lr=args.learning_rate,
                            min_lr=args.min_learning_rate,
                        ),
                    )
            # After accelerator.prepare, optimizer is an AcceleratedOptimizer:
            # on non-sync accumulation micro-steps these calls do not update/clear grads.
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        step += 1
        if first_step_pending:
            log_startup_timing(
                "first synchronized training step",
                first_step_started_at,
                accelerator=accelerator,
            )
            log_startup_timing(
                "total process startup to first step",
                process_started_at,
                accelerator=accelerator,
                synchronize_cuda=False,
            )
            first_step_pending = False
        if accelerator.is_main_process and progress is not None:
            progress.update(1)
        should_log = step == 1 or step % args.progress_every == 0
        if should_log:
            gathered = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            now = time.perf_counter()
            speed = args.progress_every / max(now - last_time, 1.0e-6)
            last_time = now
            row = {
                "loss": float(gathered),
                "loss_latent": float(args.latent_loss_weight * loss_latent.detach().float().item()),
                "loss_latent_unweighted": float(
                    args.latent_loss_weight
                    * loss_latent_unweighted.detach().float().item()
                ),
                "loss_keypoints": float(args.keypoint_loss_weight * loss_kp.detach().float().item()),
                "loss_keypoints_xy": float(loss_kp_xy.detach().float().item()),
                "loss_keypoints_z": float(loss_kp_z.detach().float().item()),
                "loss_keypoints_validity": float(
                    loss_kp_validity.detach().float().item()
                ),
                "keypoint_latent_cond_t": float(condition_latent_timestep.mean().item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "steps_per_sec": float(speed),
            }
            if not args.train_action_only:
                gathered_counts = accelerator.gather(kp_valid_counts.reshape(1, -1)).sum(dim=0)
                tp, fp, fn, target_count, pred_count, total_count = gathered_counts.tolist()
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0)
                f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
                row.update(
                    {
                        "keypoint_valid_target_rate": target_count / max(total_count, 1.0),
                        "keypoint_valid_pred_rate": pred_count / max(total_count, 1.0),
                        "keypoint_valid_precision": precision,
                        "keypoint_valid_recall": recall,
                        "keypoint_valid_f1": f1,
                    }
                )
                if text_prompt_cache is not None:
                    row.update(
                        {
                            "text_cache_size": float(
                                len(text_prompt_cache.embeddings)
                            ),
                            "text_cache_hit_rate": text_prompt_cache.hit_rate,
                        }
                    )

            for group_index, group in enumerate(optimizer.param_groups):
                group_name = str(group.get("group_name", f"group_{group_index}"))
                row[f"lr_{group_name}"] = float(group["lr"])
            if predict_action:
                row.update(
                    {
                        "loss_action": float(args.action_loss_weight * loss_action.detach().float().item()),
                        "action_latent_cond_t": float(action_condition_latent_timestep.mean().item()),
                        "action_keypoint_cond_t": float(action_condition_keypoint_timestep.mean().item()),
                    }
                )
            if cotrain_source_counts is not None:
                gathered_source_counts = accelerator.gather(
                    cotrain_source_counts.reshape(1, -1)
                ).sum(dim=0)
                total_source_count = gathered_source_counts.sum().clamp_min(1.0)
                for source_id, source_name in enumerate(cotrain_source_names):
                    source_count = gathered_source_counts[source_id]
                    row[f"data/{source_name}_samples"] = float(source_count.item())
                    row[f"data/{source_name}_fraction"] = float(
                        (source_count / total_source_count).item()
                    )
                cotrain_source_counts.zero_()
            if accelerator.is_main_process:
                if progress is not None:
                    progress.set_postfix({k: f"{v:.4g}" for k, v in row.items() if k.startswith("loss") or k == "lr"})
                if wandb_run is not None:
                    wandb_run.log({f"train/{k}": v for k, v in row.items()}, step=step)

        if args.save_step_every > 0 and step % args.save_step_every == 0:
            save_checkpoint(
                output_dir=output_dir / "checkpoints",
                step=step,
                model=model,
                optimizer=optimizer,
                accelerator=accelerator,
                args=args,
            )
            accelerator.wait_for_everyone()

    save_checkpoint(
        output_dir=output_dir / "checkpoints",
        step=step,
        model=model,
        optimizer=optimizer,
        accelerator=accelerator,
        args=args,
    )
    if progress is not None:
        progress.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
