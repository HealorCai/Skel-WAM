from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from tqdm.auto import tqdm

from skel_wam.utils.compatibility import (
    validate_fixed_training_settings,
)
from skel_wam.datasets.skel_wam_dataset import (
    SkelWamVideoKeypointDataset,
    SkelWamVideoKeypointQposDataset,
)
from skel_wam.models.skel_wam import (
    SkelWamBackboneModel,
    decode_keypoint_validity,
    infer_action_patch_embedding_mode,
    load_component_state_dict,
    normalize_keypoint_validity,
)
from skel_wam.models.wan.modules.t5 import T5EncoderModel
from skel_wam.models.wan.modules.vae2_2 import Wan2_2_VAE
from skel_wam.models.wan.utils.utils import save_video
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
from skel_wam.utils.video_metrics import video_psnr, video_ssim
from skel_wam.utils.action_metrics import real_robot_action_metrics


DEFAULT_DATASET_ROOT = REPO_ROOT / "data/EgoVLA_SIM/skeleton_video_384x384"
DEFAULT_KEYPOINT_ROOT = REPO_ROOT / "data/EgoVLA_SIM/skeleton_keypoints"
DEFAULT_ACTION_ROOT = REPO_ROOT / "data/EgoVLA_SIM/qpos"
DEFAULT_WAN_DIR = REPO_ROOT / "checkpoints/Wan-AI/Wan2.2-TI2V-5B"
CENTURY_GOTHIC_DIR = REPO_ROOT.parent / "projects/fonts"

HAND_EDGES = tuple(
    edge
    for finger in (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )
    for edge in ((0, finger[0]), *zip(finger[:-1], finger[1:]))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU multi-step inference for a Skel-WAM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--action_checkpoint",
        default=None,
        help="Optional portable Action-only checkpoint loaded after the expert backbone.",
    )
    parser.add_argument(
        "--dataset_type",
        default="skel_wam",
        choices=["skel_wam"],
        help="Dataset implementation used to construct the requested test sample.",
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--keypoint_root", default=str(DEFAULT_KEYPOINT_ROOT))
    parser.add_argument("--action_root", default=str(DEFAULT_ACTION_ROOT))
    parser.add_argument("--keypoint_dataset_key", default="keypoints")
    parser.add_argument("--action_dataset_key", default="observations/qpos")
    parser.add_argument(
        "--action_exclude_indices",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Zero-based raw qpos dimensions excluded from Action evaluation. "
            "Defaults to the mapping saved in the checkpoint."
        ),
    )
    parser.add_argument("--keypoint_stats_path", default=None)
    parser.add_argument("--action_stats_path", default=None)
    parser.add_argument("--metadata_cache_path", default=None)
    parser.add_argument("--task_include", nargs="*", default=None)
    parser.add_argument("--task_exclude", nargs="*", default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="output/test_skel_wam")

    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--global_sample_stride", type=int, default=None)
    parser.add_argument("--action_video_freq_ratio", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--history_keypoint_offsets", type=int, nargs="+", default=None)
    parser.add_argument("--action_target_frames", type=int, default=None)
    parser.add_argument("--decord_num_threads", type=int, default=1)

    parser.add_argument("--wan_checkpoint_dir", default=str(DEFAULT_WAN_DIR))
    parser.add_argument("--transformer_checkpoint_dir", default=None)
    parser.add_argument("--precomputed_text_embeddings_path", default=None)
    parser.add_argument("--t5_checkpoint_path", default=None)
    parser.add_argument("--t5_tokenizer_path", default=None)
    parser.add_argument(
        "--text_encoder_device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Used only when no precomputed embedding file is available.",
    )
    parser.add_argument("--prompt_template", default=None)
    parser.add_argument("--empty_task_name", default=None)
    parser.add_argument(
        "--task_lang_map_path",
        default=None,
        help="Optional JSON mapping from dataset task names to task instructions.",
    )
    parser.add_argument(
        "--disable_task_lang_map",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--latent_num_inference_steps", type=int, default=None)
    parser.add_argument("--keypoint_num_inference_steps", type=int, default=None)
    parser.add_argument("--action_num_inference_steps", type=int, default=None)
    parser.add_argument("--latent_snr_shift", type=float, default=None)
    parser.add_argument("--keypoint_snr_shift", type=float, default=None)
    parser.add_argument("--action_snr_shift", type=float, default=None)
    parser.add_argument(
        "--keypoint_video_condition",
        default="pred",
        choices=["pred", "gt"],
    )
    parser.add_argument(
        "--action_condition",
        default="pred",
        choices=["pred", "gt"],
    )
    parser.add_argument(
        "--run_action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vis_keypoint_video_attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Capture and visualize actual Kn-query to Vc-key attention at "
            "high/mid/low keypoint denoising stages."
        ),
    )
    parser.add_argument(
        "--vis_keypoint_video_same_frame_attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Visualize only each Kn query frame's time-aligned Vc latent frame, "
            "without summing or interpolating attention across time."
        ),
    )
    parser.add_argument(
        "--vis_action_modality_contribution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Measure and visualize Action-head Video/Keypoint contribution "
            "shares from their gated residual-update RMS at high-, mid-, and "
            "low-noise Action denoising stages."
        ),
    )
    parser.add_argument(
        "--attn_vis_keep_height_ratio",
        type=float,
        default=1.0,
        help=(
            "Top-of-frame height ratio retained in Kn->Vc attention videos. "
            "Cropping is aligned to whole latent-attention rows and the retained "
            "attention is renormalized."
        ),
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--overlay_radius", type=int, default=3)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def inference_autocast(tensor: torch.Tensor):
    """Match the mixed-precision context used by the training wrapper."""
    enabled = tensor.is_cuda and tensor.dtype in (
        torch.float16,
        torch.bfloat16,
    )
    return torch.autocast(
        device_type=tensor.device.type,
        dtype=tensor.dtype,
        enabled=enabled,
    )


def checkpoint_value(
    args: argparse.Namespace,
    checkpoint_args: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    checkpoint_value_or_none = checkpoint_args.get(name)
    return default if checkpoint_value_or_none is None else checkpoint_value_or_none


def load_checkpoint(path: str) -> dict[str, Any]:
    payload = torch.load(
        str(Path(path).expanduser()),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid checkpoint payload: {type(payload)}")
    return payload


def encode_video(
    vae: Wan2_2_VAE,
    video: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        encoded = vae.encode([video.to(device=device, dtype=dtype)])
    if encoded is None:
        raise RuntimeError("Wan VAE encode returned None")
    return encoded[0].unsqueeze(0).to(device=device, dtype=dtype)


def decode_video(vae: Wan2_2_VAE, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        decoded = vae.decode([latents[0].detach().float()])
    if decoded is None:
        raise RuntimeError("Wan VAE decode returned None")
    return decoded[0].detach().float().cpu()


def build_text_condition(
    prompts: list[str],
    *,
    precomputed_path: str | None,
    wan_checkpoint_dir: Path,
    t5_checkpoint_path: str | None,
    t5_tokenizer_path: str | None,
    text_encoder_device: str,
    output_device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    if precomputed_path:
        embeddings, _ = load_precomputed_text_embeddings(precomputed_path)
        return PrecomputedPromptEmbeddingStore(
            embeddings, expected_text_dim=4096
        ).encode(
            prompts,
            output_device=output_device,
            output_dtype=dtype,
        )
    encoder_device = torch.device(text_encoder_device)
    if encoder_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA T5 encoding was requested but CUDA is unavailable")
    encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=encoder_device,
        checkpoint_path=str(
            Path(t5_checkpoint_path)
            if t5_checkpoint_path
            else wan_checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        ),
        tokenizer_path=str(
            Path(t5_tokenizer_path)
            if t5_tokenizer_path
            else wan_checkpoint_dir / "google/umt5-xxl"
        ),
        shard_fn=None,
    )
    encoder.model.eval().requires_grad_(False)
    with torch.no_grad():
        return [
            embedding.to(device=output_device, dtype=dtype)
            for embedding in encoder(prompts, encoder_device)
        ]


@torch.no_grad()
def sample_video_latents(
    model: SkelWamBackboneModel,
    scheduler: FlowMatchingScheduler,
    *,
    condition_latents: torch.Tensor,
    history_keypoints: torch.Tensor | None,
    current_keypoints: torch.Tensor | None,
    text_emb: Sequence[torch.Tensor],
    history_keypoints_valid: torch.Tensor | None,
    current_keypoints_valid: torch.Tensor | None,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=condition_latents.device).manual_seed(seed)
    latents = torch.randn(
        condition_latents.shape,
        generator=generator,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    latents[:, :, 0:1] = condition_latents[:, :, 0:1]
    history_cache = None
    if model.use_keypoint:
        if history_keypoints is None or current_keypoints is None:
            raise ValueError("Keypoint-enabled video sampling requires keypoint history")
        with inference_autocast(condition_latents):
            history_cache = model.prefill_history_keypoint_cache(
                history_keypoints=history_keypoints,
                current_keypoints=current_keypoints,
                text_emb=text_emb,
                history_keypoints_valid=history_keypoints_valid,
                current_keypoints_valid=current_keypoints_valid,
            )
    for timestep in tqdm(
        scheduler.timesteps.to(device=latents.device, dtype=latents.dtype),
        desc="sample video",
        dynamic_ncols=True,
    ):
        timestep_batch = timestep.expand(latents.shape[0])
        with inference_autocast(latents):
            if model.use_keypoint:
                velocity = model.forward_latent_step(
                    noisy_latents=latents,
                    condition_latents=condition_latents,
                    latent_timestep=timestep_batch,
                    text_emb=text_emb,
                    history_keypoint_cache=history_cache,
                )["latent_velocity"]
            else:
                velocity = model.forward_video_only(
                    noisy_latents=latents,
                    condition_latents=condition_latents,
                    latent_timestep=timestep_batch,
                    text_emb=text_emb,
                )["latent_velocity"]
        latents = scheduler.step(velocity, timestep, latents)
        latents[:, :, 0:1] = condition_latents[:, :, 0:1]
    return latents


@torch.no_grad()
def sample_keypoint_state(
    model: SkelWamBackboneModel,
    scheduler: FlowMatchingScheduler,
    *,
    condition_latents: torch.Tensor,
    current_keypoints: torch.Tensor,
    current_keypoints_valid: torch.Tensor,
    text_emb: Sequence[torch.Tensor],
    seed: int,
    capture_attention: bool = False,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]] | None]:
    expert = model.keypoint_expert
    if expert is None:
        raise RuntimeError("Keypoint sampling is unavailable when use_keypoint=False")
    generator = torch.Generator(device=condition_latents.device).manual_seed(seed)
    state = torch.randn(
        condition_latents.shape[0],
        model.keypoint_frames,
        expert.num_hands,
        expert.num_keypoints,
        expert.keypoint_dims + 1,
        generator=generator,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    current_state = torch.cat(
        [
            current_keypoints,
            normalize_keypoint_validity(
                current_keypoints_valid,
                dtype=current_keypoints.dtype,
            ).unsqueeze(-1),
        ],
        dim=-1,
    )
    state[:, 0:1] = current_state
    with inference_autocast(condition_latents):
        video_cache = model.prefill_condition_video_cache(
            condition_latents=condition_latents,
            text_emb=text_emb,
        )
    attention_stages: dict[str, dict[str, Any]] | None = (
        {} if capture_attention else None
    )
    final_step_index = int(scheduler.timesteps.numel()) - 1
    capture_stage_by_index = {
        0: "high_noise",
        final_step_index // 2: "mid_noise",
        final_step_index: "low_noise",
    }
    for step_index, timestep in enumerate(
        tqdm(
            scheduler.timesteps.to(device=state.device, dtype=state.dtype),
            desc="sample keypoint",
            dynamic_ncols=True,
        )
    ):
        timestep_batch = timestep.expand(state.shape[0])
        capture_stage = capture_stage_by_index.get(step_index)
        with inference_autocast(state):
            output = model.forward_keypoint_step(
                noisy_keypoints=state[..., : expert.keypoint_dims],
                noisy_keypoint_validity=state[..., expert.keypoint_dims],
                current_keypoints=current_keypoints,
                current_keypoints_valid=current_keypoints_valid,
                keypoint_timestep=timestep_batch,
                text_emb=text_emb,
                condition_video_cache=video_cache,
                capture_attention=(
                    capture_attention and capture_stage is not None
                ),
            )
        if "kn_vc_attention" in output:
            if attention_stages is None or capture_stage is None:
                raise RuntimeError("Unexpected Kn->Vc attention capture")
            attention_stages[capture_stage] = {
                "step_index": step_index,
                "timestep": float(timestep.detach().float().cpu().item()),
                "attention": output["kn_vc_attention"],
                "grid_size": tuple(
                    int(value)
                    for value in output[
                        "kn_vc_attention_grid_size"
                    ].tolist()
                ),
            }
        state = scheduler.step(
            output["keypoint_flow_velocity"], timestep, state
        )
        state[:, 0:1] = current_state
    return state, attention_stages


@torch.no_grad()
def sample_action(
    model: SkelWamBackboneModel,
    scheduler: FlowMatchingScheduler,
    *,
    condition_latents: torch.Tensor,
    condition_keypoints: torch.Tensor | None,
    condition_keypoints_valid: torch.Tensor | None,
    seed: int,
    rtc_prefix: torch.Tensor | None = None,
    rtc_weights: torch.Tensor | None = None,
    capture_modality_contributions: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    generator = torch.Generator(device=condition_latents.device).manual_seed(seed)
    action = torch.randn(
        condition_latents.shape[0],
        model.action_frames,
        model.action_dim,
        generator=generator,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    initial_noise = action.clone()
    if (rtc_prefix is None) != (rtc_weights is None):
        raise ValueError("rtc_prefix and rtc_weights must be provided together")
    if rtc_prefix is not None:
        rtc_prefix = rtc_prefix.to(device=action.device, dtype=action.dtype)
        rtc_weights = rtc_weights.to(device=action.device, dtype=action.dtype)
        if rtc_prefix.shape != action.shape:
            raise ValueError(
                f"rtc_prefix must have shape {tuple(action.shape)}, "
                f"got {tuple(rtc_prefix.shape)}"
            )
        if rtc_weights.shape != (action.shape[0], action.shape[1], 1):
            raise ValueError(
                "rtc_weights must have shape [B,H,1], got "
                f"{tuple(rtc_weights.shape)}"
            )
        if not bool(torch.all((rtc_weights >= 0) & (rtc_weights <= 1))):
            raise ValueError("rtc_weights must be in [0,1]")
    contribution_stages: dict[str, dict[str, Any]] = {}
    final_step_index = int(scheduler.timesteps.numel()) - 1
    capture_stage_by_index = {
        0: "high_noise",
        final_step_index // 2: "mid_noise",
        final_step_index: "low_noise",
    }
    for step_index, timestep in enumerate(tqdm(
        scheduler.timesteps.to(device=action.device, dtype=action.dtype),
        desc="sample action",
        dynamic_ncols=True,
    )):
        timestep_batch = timestep.expand(action.shape[0])
        capture_stage = capture_stage_by_index.get(step_index)
        with inference_autocast(action):
            output = model.forward_action_fm(
                condition_latents=condition_latents,
                condition_keypoints=condition_keypoints,
                condition_keypoints_valid=condition_keypoints_valid,
                noisy_action=action,
                action_timestep=timestep_batch,
                capture_modality_contributions=(
                    capture_modality_contributions and capture_stage is not None
                ),
            )
        velocity = output["action_velocity"]
        if "action_modality_contributions" in output:
            if capture_stage is None:
                raise RuntimeError("Unexpected Action modality contribution capture")
            contribution_stages[capture_stage] = {
                "step_index": step_index,
                "timestep": float(timestep.detach().float().cpu().item()),
                **{
                    name: value.detach().float().cpu()
                    for name, value in output[
                        "action_modality_contributions"
                    ].items()
                },
            }
        action = scheduler.step(velocity, timestep, action)
        if rtc_prefix is not None and rtc_weights is not None:
            delta = scheduler.deltas[step_index].to(
                device=action.device, dtype=action.dtype
            )
            next_timestep = (timestep + delta).clamp(0.0, 1.0)
            known_sample = (
                (1.0 - next_timestep) * rtc_prefix
                + next_timestep * initial_noise
            )
            action = (
                rtc_weights * known_sample
                + (1.0 - rtc_weights) * action
            )
    if capture_modality_contributions:
        return action, contribution_stages
    return action


def video_to_unit(video: torch.Tensor) -> torch.Tensor:
    return ((video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5)


def video_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    video = (
        video.detach().float().cpu().clamp(-1.0, 1.0).add(1.0).mul(127.5)
        .round().to(torch.uint8)
    )
    return [
        video[:, index].permute(1, 2, 0).contiguous().numpy()
        for index in range(video.shape[1])
    ]


def save_frames(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def save_action_modality_contribution_visualizations(
    contribution_stages: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Save gated residual-RMS shares for Action Video/Keypoint branches."""
    stage_order = ["high_noise", "mid_noise", "low_noise"]
    missing_stages = [
        stage for stage in stage_order if stage not in contribution_stages
    ]
    if missing_stages:
        raise ValueError(
            "Missing Action modality contribution stages: "
            + ", ".join(missing_stages)
        )

    stage_summaries: dict[str, dict[str, Any]] = {}
    for stage_name in stage_order:
        stage = contribution_stages[stage_name]
        video_rms = stage["video_rms"]
        keypoint_rms = stage["keypoint_rms"]
        if video_rms.ndim != 3 or video_rms.shape != keypoint_rms.shape:
            raise ValueError(
                "Expected matching Action contribution tensors [B,layer,frame], "
                f"got {tuple(video_rms.shape)} and {tuple(keypoint_rms.shape)}"
            )
        # Layers contribute successive residual updates. Sum their RMS strength
        # first, compute the modality share per Action frame, then take the
        # requested arithmetic mean over batch items and Action frames.
        video_strength = video_rms.sum(dim=1)
        keypoint_strength = keypoint_rms.sum(dim=1)
        frame_denominator = (video_strength + keypoint_strength).clamp_min(1.0e-12)
        video_frame_share = video_strength / frame_denominator
        keypoint_frame_share = keypoint_strength / frame_denominator
        stage_summaries[stage_name] = {
            "step_index": int(stage["step_index"]),
            "timestep": float(stage["timestep"]),
            "video_share": float(video_frame_share.mean().item()),
            "keypoint_share": float(keypoint_frame_share.mean().item()),
            "aggregation": "sum layers, then mean per-frame share",
            "keypoint_gate_by_layer": [
                float(value) for value in stage["keypoint_gate"].flatten().tolist()
            ],
        }
    raw_path = output_dir / "action_modality_contribution.pt"
    torch.save(
        {
            "stages": contribution_stages,
            "shape": "[batch, action_layer, action_frame]",
            "metric": "RMS of each branch's actual residual update",
            "video_delta": "post-output-projection and post-dropout",
            "keypoint_delta": "post-output-projection, sigmoid gate, and post-dropout",
            "share_aggregation": "sum layer RMS, compute per-frame share, then mean over batch and Action frames",
            "interpretation": "local residual contribution proxy, not shared-softmax attention mass",
        },
        raw_path,
    )

    share_path = output_dir / "action_modality_contribution_share.png"
    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    video_color = "#4F46E5"
    keypoint_color = "#F97316"
    stage_labels = {
        "high_noise": "High Noise",
        "mid_noise": "Mid Noise",
        "low_noise": "Low Noise",
    }
    y_positions = np.arange(len(stage_order))[::-1]
    keypoint_means = np.array(
        [stage_summaries[name]["keypoint_share"] for name in stage_order]
    )
    video_means = np.array(
        [stage_summaries[name]["video_share"] for name in stage_order]
    )
    bar_height = 0.56
    axis.barh(
        y_positions,
        -keypoint_means,
        height=bar_height,
        color=keypoint_color,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.0,
    )
    axis.barh(
        y_positions,
        video_means,
        height=bar_height,
        color=video_color,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.0,
    )
    for y, keypoint_mean, video_mean in zip(
        y_positions, keypoint_means, video_means
    ):
        axis.text(
            -keypoint_mean + 0.025,
            y,
            f"{keypoint_mean:.1%}",
            ha="left",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
        axis.text(
            video_mean - 0.025,
            y,
            f"{video_mean:.1%}",
            ha="right",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
    axis.axvline(0.0, color="#334155", linewidth=1.4, zorder=4)
    axis.set_title(
        "Action Modality Contribution Across Noise Levels",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        pad=24,
    )
    axis.text(
        0.0,
        1.025,
        "Mean Gated Residual RMS Share Across Action Frames",
        transform=axis.transAxes,
        fontsize=10.5,
        color="#64748B",
        va="bottom",
    )
    axis.text(
        0.25,
        0.93,
        "←  Keypoint",
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=keypoint_color,
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        0.75,
        0.93,
        "Video Latent  →",
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=video_color,
        fontsize=12,
        fontweight="bold",
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [stage_labels[name] for name in stage_order],
        fontsize=11,
        fontweight="semibold",
    )
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-0.7, len(stage_order) - 0.25)
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"{abs(value):.0%}")
    )
    axis.set_xlabel(
        "Mean Modality Contribution Share",
        fontsize=11,
        color="#334155",
        labelpad=12,
    )
    axis.grid(axis="x", color="#CBD5E1", alpha=0.6, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", colors="#475569", labelsize=9.5, length=0)
    axis.tick_params(axis="y", colors="#334155", length=0, pad=10)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)
    for font_path in sorted(CENTURY_GOTHIC_DIR.glob("Century Gothic*.ttf")):
        font_manager.fontManager.addfont(font_path)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chart_font = (
        "Century Gothic"
        if "Century Gothic" in available_fonts
        else "DejaVu Sans"
    )
    for text_artist in figure.findobj(match=matplotlib.text.Text):
        text_artist.set_fontfamily(chart_font)
    figure.subplots_adjust(left=0.16, right=0.98, top=0.8, bottom=0.16)
    figure.savefig(share_path, dpi=180)
    plt.close(figure)

    overall_video_share = float(video_means.mean())
    overall_keypoint_share = float(keypoint_means.mean())
    return {
        "action_modality_contribution_tensor_path": str(raw_path),
        "action_modality_contribution_share_path": str(share_path),
        "action_modality_contribution_metric": "gated residual RMS share",
        "action_modality_contribution_overall": {
            "video_share": overall_video_share,
            "keypoint_share": overall_keypoint_share,
            "aggregation": "mean of high/mid/low stage mean per-frame shares",
        },
        "action_modality_contribution_by_stage": stage_summaries,
    }


def save_kn_vc_attention_visualizations(
    attention_stages: dict[str, dict[str, Any]],
    *,
    condition_frames: list[np.ndarray],
    num_hands: int,
    output_dir: Path,
    fps: int,
    keep_height_ratio: float = 1.0,
    save_overlay_videos: bool = True,
    overlay_callback: Callable[[str, list[np.ndarray]], None] | None = None,
) -> dict[str, Any]:
    """Save actual shared-softmax Kn-query to Vc-key attention maps."""
    stage_order = [
        stage
        for stage in ("high_noise", "mid_noise", "low_noise")
        if stage in attention_stages
    ]
    if not stage_order:
        raise ValueError("No Kn->Vc attention stages were captured")
    hand_names = (
        ("right", "left")
        if num_hands == 2
        else tuple(f"hand_{index}" for index in range(num_hands))
    )
    components: dict[str, dict[str, torch.Tensor | tuple[int, int, int]]] = {}
    expected_grid: tuple[int, int, int] | None = None
    for stage_name in stage_order:
        stage = attention_stages[stage_name]
        attention = stage["attention"].detach().float().cpu()
        grid = tuple(int(value) for value in stage["grid_size"])
        if expected_grid is None:
            expected_grid = grid
        elif grid != expected_grid:
            raise ValueError(
                f"Attention grid changed across stages: {expected_grid} vs {grid}"
            )
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(
                "Expected Kn->Vc attention [1,layers,queries,condition_tokens], "
                f"got {tuple(attention.shape)}"
            )
        condition_tokens = int(np.prod(grid))
        if attention.shape[-1] != condition_tokens:
            raise ValueError(
                f"Expected {condition_tokens} Vc keys, got {attention.shape[-1]}"
            )
        if attention.shape[-2] % num_hands:
            raise ValueError(
                f"Query count {attention.shape[-2]} is not divisible by {num_hands}"
            )
        query_frames = attention.shape[-2] // num_hands
        layer_attention = attention[0].reshape(
            attention.shape[1],
            query_frames,
            num_hands,
            *grid,
        )
        layer_mean = layer_attention.mean(dim=0)
        keep_rows = round(grid[1] * keep_height_ratio)
        if not 0 < keep_rows <= grid[1]:
            raise ValueError(
                "attn_vis_keep_height_ratio must retain at least one and at most "
                f"all latent rows, got ratio={keep_height_ratio} for height={grid[1]}"
            )
        layer_mean = layer_mean[..., :keep_rows, :]
        cross_mass = layer_mean.sum(dim=(2, 3, 4))
        condition_normalized = layer_mean / cross_mass[
            ..., None, None, None
        ].clamp_min(1.0e-12)
        components[stage_name] = {
            "grid": (grid[0], keep_rows, grid[2]),
            "cross_mass": cross_mass,
            "spatial": condition_normalized.sum(dim=2),
            "temporal": condition_normalized.sum(dim=(3, 4)),
            "layer_cross_mass": layer_attention.sum(dim=(3, 4, 5)),
        }

    raw_path = output_dir / "kn_vc_attention.pt"
    torch.save(
        {
            "stages": attention_stages,
            "shape": "[batch, layer, keypoint_frame*hand, Vc_token]",
            "grid_size": expected_grid,
            "query_order": "frame-major, then hand",
            "hand_order": list(hand_names),
            "head_aggregation": "mean",
            "normalization": "shared softmax over Vc and Kn key segments",
            "rope": "temporal-only Kn-to-Vc cross RoPE",
        },
        raw_path,
    )

    overlay_paths: dict[str, dict[str, str]] = {}
    video_height, video_width = condition_frames[0].shape[:2]
    keep_rows = int(components[stage_order[0]]["grid"][1])
    crop_height = round(video_height * keep_rows / int(expected_grid[1]))
    for stage_name in stage_order:
        component = components[stage_name]
        spatial = component["spatial"]
        overlay_paths[stage_name] = {}
        for hand_index, hand_name in enumerate(hand_names):
            spatial_hand = spatial[:, hand_index]
            spatial_video = torch.nn.functional.interpolate(
                spatial_hand[None, None],
                size=(len(condition_frames), crop_height, video_width),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
            scale = max(
                float(torch.quantile(spatial_video.flatten(), 0.99).item()),
                1.0e-12,
            )
            heat = (spatial_video / scale).clamp(0.0, 1.0)
            frames = []
            for frame, frame_heat in zip(condition_frames, heat):
                heat_u8 = (
                    (frame_heat.numpy() * 255.0).round().astype(np.uint8)
                )
                heat_rgb = cv2.cvtColor(
                    cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET),
                    cv2.COLOR_BGR2RGB,
                )
                overlay = cv2.addWeighted(
                    np.ascontiguousarray(frame[:crop_height]),
                    0.58,
                    heat_rgb,
                    0.42,
                    0.0,
                )
                frames.append(overlay)
            stream_name = f"kn_vc_attention_{stage_name}_{hand_name}"
            if overlay_callback is not None:
                overlay_callback(stream_name, frames)
            if save_overlay_videos:
                path = output_dir / f"{stream_name}.mp4"
                save_frames(path, frames, fps=fps)
                overlay_paths[stage_name][hand_name] = str(path)

    temporal_path = output_dir / "kn_vc_attention_temporal.png"
    figure, axes = plt.subplots(
        len(stage_order),
        num_hands,
        figsize=(5.2 * num_hands, 3.2 * len(stage_order)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    temporal_image = None
    for row, stage_name in enumerate(stage_order):
        temporal = components[stage_name]["temporal"]
        for hand_index, hand_name in enumerate(hand_names):
            axis = axes[row][hand_index]
            temporal_image = axis.imshow(
                temporal[:, hand_index].numpy(),
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_title(f"{stage_name}: {hand_name}")
            axis.set_xlabel("Vc latent frame")
            axis.set_ylabel("Kn query frame")
    figure.colorbar(
        temporal_image,
        ax=axes.ravel().tolist(),
        label="Vc-conditional attention",
        shrink=0.85,
    )
    figure.subplots_adjust(right=0.88, hspace=0.35, wspace=0.2)
    figure.savefig(temporal_path, dpi=180)
    plt.close(figure)

    layer_mass_path = output_dir / "kn_vc_attention_layer_mass.png"
    figure, axes = plt.subplots(
        1,
        num_hands,
        figsize=(6.0 * num_hands, 4.0),
        squeeze=False,
        sharey=True,
    )
    for hand_index, hand_name in enumerate(hand_names):
        axis = axes[0][hand_index]
        for stage_name in stage_order:
            layer_mass = components[stage_name]["layer_cross_mass"][
                :, :, hand_index
            ].mean(dim=1)
            axis.plot(
                np.arange(layer_mass.shape[0]),
                layer_mass.numpy(),
                label=stage_name,
            )
        axis.set_title(hand_name)
        axis.set_xlabel("expert layer")
        axis.set_ylabel("mean Kn->Vc attention mass")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(layer_mass_path, dpi=180)
    plt.close(figure)

    return {
        "kn_vc_attention_tensor_path": str(raw_path),
        "kn_vc_attention_overlay_paths": overlay_paths,
        "kn_vc_attention_temporal_path": str(temporal_path),
        "kn_vc_attention_layer_mass_path": str(layer_mass_path),
        "kn_vc_attention_grid_size": list(expected_grid or ()),
        "kn_vc_attention_visible_grid_size": [
            int(expected_grid[0]), keep_rows, int(expected_grid[2])
        ],
        "kn_vc_attention_output_size": [crop_height, video_width],
        "kn_vc_attention_hand_order": list(hand_names),
    }


def _aligned_time_indices(source_frames: int, target_frames: int) -> torch.Tensor:
    """Map source-frame centers to the nearest target-frame centers."""
    if source_frames <= 0 or target_frames <= 0:
        raise ValueError(
            f"Frame counts must be positive, got {source_frames} and {target_frames}"
        )
    if source_frames == 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.linspace(0, target_frames - 1, source_frames).round().long()


def save_kn_vc_same_frame_attention_visualizations(
    attention_stages: dict[str, dict[str, Any]],
    *,
    condition_frames: list[np.ndarray],
    num_hands: int,
    output_dir: Path,
    fps: int,
    keep_height_ratio: float = 1.0,
    save_overlay_videos: bool = True,
    overlay_callback: Callable[[str, list[np.ndarray]], None] | None = None,
) -> dict[str, Any]:
    """Save Kn frame i -> time-aligned Vc latent frame attention maps."""
    if not condition_frames:
        raise ValueError("condition_frames must not be empty")
    stage_order = [
        stage
        for stage in ("high_noise", "mid_noise", "low_noise")
        if stage in attention_stages
    ]
    if not stage_order:
        raise ValueError("No Kn->Vc attention stages were captured")
    hand_names = (
        ("right", "left")
        if num_hands == 2
        else tuple(f"hand_{index}" for index in range(num_hands))
    )
    video_height, video_width = condition_frames[0].shape[:2]
    overlay_paths: dict[str, dict[str, str]] = {}
    time_mappings: dict[str, dict[str, list[int]]] = {}

    for stage_name in stage_order:
        stage = attention_stages[stage_name]
        attention = stage["attention"].detach().float().cpu()
        grid = tuple(int(value) for value in stage["grid_size"])
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(
                "Expected Kn->Vc attention [1,layers,queries,condition_tokens], "
                f"got {tuple(attention.shape)}"
            )
        if attention.shape[-1] != int(np.prod(grid)):
            raise ValueError(
                f"Attention has {attention.shape[-1]} Vc keys for grid {grid}"
            )
        if attention.shape[-2] % num_hands:
            raise ValueError(
                f"Query count {attention.shape[-2]} is not divisible by {num_hands}"
            )

        query_frames = attention.shape[-2] // num_hands
        layer_mean = attention[0].reshape(
            attention.shape[1], query_frames, num_hands, *grid
        ).mean(dim=0)
        keep_rows = round(grid[1] * keep_height_ratio)
        if not 0 < keep_rows <= grid[1]:
            raise ValueError(
                "attn_vis_keep_height_ratio must retain at least one and at most "
                f"all latent rows, got ratio={keep_height_ratio} for height={grid[1]}"
            )
        layer_mean = layer_mean[..., :keep_rows, :]
        crop_height = round(video_height * keep_rows / grid[1])
        cross_mass = layer_mean.sum(dim=(2, 3, 4))
        condition_normalized = layer_mean / cross_mass[
            ..., None, None, None
        ].clamp_min(1.0e-12)

        vc_indices = _aligned_time_indices(query_frames, grid[0])
        image_indices = _aligned_time_indices(
            query_frames, len(condition_frames)
        )
        query_indices = torch.arange(query_frames)
        same_frame = condition_normalized[query_indices, :, vc_indices]
        time_mappings[stage_name] = {
            "kn_to_vc": vc_indices.tolist(),
            "kn_to_image": image_indices.tolist(),
        }
        overlay_paths[stage_name] = {}

        for hand_index, hand_name in enumerate(hand_names):
            spatial = same_frame[:, hand_index]
            spatial_image = torch.nn.functional.interpolate(
                spatial[:, None],
                size=(crop_height, video_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            scale = max(
                float(torch.quantile(spatial_image.flatten(), 0.99).item()),
                1.0e-12,
            )
            heat = (spatial_image / scale).clamp(0.0, 1.0)
            frames = []
            for query_index in range(query_frames):
                image_index = int(image_indices[query_index])
                heat_u8 = (
                    (heat[query_index].numpy() * 255.0)
                    .round()
                    .astype(np.uint8)
                )
                heat_rgb = cv2.cvtColor(
                    cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET),
                    cv2.COLOR_BGR2RGB,
                )
                overlay = cv2.addWeighted(
                    np.ascontiguousarray(
                        condition_frames[image_index][:crop_height]
                    ),
                    0.58,
                    heat_rgb,
                    0.42,
                    0.0,
                )
                frames.append(overlay)
            stream_name = (
                f"kn_vc_same_frame_attention_{stage_name}_{hand_name}"
            )
            if overlay_callback is not None:
                overlay_callback(stream_name, frames)
            if save_overlay_videos:
                path = output_dir / f"{stream_name}.mp4"
                save_frames(path, frames, fps=fps)
                overlay_paths[stage_name][hand_name] = str(path)

    return {
        "kn_vc_same_frame_attention_overlay_paths": overlay_paths,
        "kn_vc_same_frame_attention_time_mappings": time_mappings,
    }


def denormalize_keypoints(
    sample: dict[str, Any], keypoints: torch.Tensor
) -> torch.Tensor:
    if "keypoint_mean" not in sample:
        return keypoints
    mean = sample["keypoint_mean"].to(keypoints)
    std = sample["keypoint_std"].to(keypoints)
    shape = (1,) * (keypoints.ndim - 1) + (mean.numel(),)
    return keypoints * std.reshape(shape) + mean.reshape(shape)


def denormalize_action(sample: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
    if "qpos_mean" not in sample:
        return action
    mean = sample["qpos_mean"].to(action)
    std = sample["qpos_std"].to(action)
    return action * std + mean


def normalized_to_pixel(
    point: np.ndarray, width: int, height: int
) -> tuple[int, int] | None:
    if not np.isfinite(point[:2]).all() or np.any(point[:2] < 0):
        return None
    x, y = float(point[0]), float(point[1])
    if x <= 1.5 and y <= 1.5:
        x *= width - 1
        y *= height - 1
    return (
        int(round(np.clip(x, 0, width - 1))),
        int(round(np.clip(y, 0, height - 1))),
    )


def overlay_keypoints(
    frames: list[np.ndarray],
    keypoints: torch.Tensor,
    validity: torch.Tensor,
    *,
    color: tuple[int, int, int],
    radius: int,
) -> list[np.ndarray]:
    points = keypoints.detach().float().cpu().numpy().copy()
    points[~validity.detach().cpu().bool().numpy()] = np.nan
    output = []
    for index, frame in enumerate(frames):
        image = np.ascontiguousarray(frame.copy())
        frame_points = points[min(index, len(points) - 1)]
        for hand in frame_points:
            pixels = [
                normalized_to_pixel(point, image.shape[1], image.shape[0])
                for point in hand
            ]
            if len(pixels) == 21:
                for start, stop in HAND_EDGES:
                    if pixels[start] is not None and pixels[stop] is not None:
                        cv2.line(
                            image, pixels[start], pixels[stop], color, 1, cv2.LINE_AA
                        )
            for pixel in pixels:
                if pixel is not None:
                    cv2.circle(image, pixel, radius, color, -1, cv2.LINE_AA)
        output.append(image)
    return output


def plot_action(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    path: Path,
) -> None:
    prediction = prediction.detach().float().cpu().numpy()
    target = target.detach().float().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    dimensions = target.shape[-1]
    columns = 5
    rows = (dimensions + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2.7 * rows), squeeze=False)
    for dimension, axis in enumerate(axes.flat):
        if dimension >= dimensions:
            axis.axis("off")
            continue
        x = np.arange(target.shape[0])
        axis.plot(x[valid_np], target[valid_np, dimension], label="GT", linewidth=1.2)
        axis.plot(
            x[valid_np], prediction[valid_np, dimension], label="pred", linewidth=1.0
        )
        axis.set_title(f"action[{dimension}]")
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def classification_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.bool()
    target = target.bool()
    tp = float((prediction & target).sum().item())
    fp = float((prediction & ~target).sum().item())
    fn = float((~prediction & target).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "keypoint_valid_target_rate": float(target.float().mean().item()),
        "keypoint_valid_pred_rate": float(prediction.float().mean().item()),
        "keypoint_valid_precision": precision,
        "keypoint_valid_recall": recall,
        "keypoint_valid_f1": 2.0 * precision * recall / max(
            precision + recall, 1.0e-12
        ),
    }


def tensor_json(value: torch.Tensor | None) -> Any:
    return None if value is None else value.detach().float().cpu().tolist()


_EVAL_RUNTIME_CACHE: dict[str, Any] = {}


def clear_eval_runtime_cache() -> None:
    """Release models cached by repeated in-process evaluation calls."""
    _EVAL_RUNTIME_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_key = (
        str(Path(args.checkpoint).expanduser().resolve()),
        str(Path(args.action_checkpoint).expanduser().resolve())
        if args.action_checkpoint
        else None,
        str(device),
        str(dtype),
    )
    if _EVAL_RUNTIME_CACHE.get("key") != cache_key:
        clear_eval_runtime_cache()
        print(f"[eval-runtime] loading checkpoint once: {args.checkpoint}", flush=True)
        checkpoint_payload = load_checkpoint(args.checkpoint)
        checkpoint_args = checkpoint_payload.get("args", {})
        if not isinstance(checkpoint_args, dict):
            checkpoint_args = {}
        state_dict = checkpoint_payload.get("model", checkpoint_payload)
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain a model state dict")
        action_payload = (
            load_checkpoint(args.action_checkpoint)
            if args.action_checkpoint
            else None
        )
        action_state_dict = (
            action_payload.get("model", action_payload)
            if action_payload is not None
            else None
        )
        if action_state_dict is not None and not isinstance(action_state_dict, dict):
            raise TypeError("Action checkpoint does not contain a model state dict")
        action_checkpoint_args = (
            action_payload.get("args", {}) if action_payload is not None else {}
        )
        if not isinstance(action_checkpoint_args, dict):
            action_checkpoint_args = {}
        effective_action_state = action_state_dict or state_dict
        predicts_action = any(
            str(name).startswith("action_fm_head.")
            for name in effective_action_state
        )
        action_patch_embedding_mode = infer_action_patch_embedding_mode(
            state_dict,
            action_state=action_state_dict,
        )
        action_mode_args = (
            action_checkpoint_args if action_state_dict is not None else checkpoint_args
        )
        declared_action_patch_mode = action_mode_args.get(
            "action_patch_embedding_mode"
        )
        if (
            declared_action_patch_mode is not None
            and str(declared_action_patch_mode) != action_patch_embedding_mode
        ):
            raise ValueError(
                "Checkpoint action_patch_embedding_mode disagrees with its state: "
                f"declared={declared_action_patch_mode}, "
                f"inferred={action_patch_embedding_mode}."
            )
        action_embedder_weight = effective_action_state.get(
            "action_fm_head.action_embedder.weight"
        )
        _EVAL_RUNTIME_CACHE.update(
            {
                "key": cache_key,
                "checkpoint_step": checkpoint_payload.get("step"),
                "checkpoint_args": checkpoint_args,
                "action_checkpoint_args": action_checkpoint_args,
                "state_dict": state_dict,
                "action_state_dict": action_state_dict,
                "predicts_action": predicts_action,
                "action_patch_embedding_mode": action_patch_embedding_mode,
                "checkpoint_action_dim": (
                    int(action_embedder_weight.shape[1])
                    if isinstance(action_embedder_weight, torch.Tensor)
                    else None
                ),
            }
        )
    checkpoint_step = _EVAL_RUNTIME_CACHE["checkpoint_step"]
    checkpoint_args = _EVAL_RUNTIME_CACHE["checkpoint_args"]
    action_checkpoint_args = _EVAL_RUNTIME_CACHE["action_checkpoint_args"]
    state_dict = _EVAL_RUNTIME_CACHE.get("state_dict")
    action_state_dict = _EVAL_RUNTIME_CACHE.get("action_state_dict")
    predicts_action = bool(_EVAL_RUNTIME_CACHE["predicts_action"])
    evaluate_action = bool(args.run_action and predicts_action)
    action_patch_embedding_mode = str(
        _EVAL_RUNTIME_CACHE["action_patch_embedding_mode"]
    )

    num_frames = int(checkpoint_value(args, checkpoint_args, "num_frames", 33))
    stride = int(
        checkpoint_value(args, checkpoint_args, "global_sample_stride", 1)
    )
    ratio = int(
        checkpoint_value(args, checkpoint_args, "action_video_freq_ratio", 4)
    )
    image_size = tuple(
        int(value)
        for value in checkpoint_value(args, checkpoint_args, "image_size", [384, 384])
    )
    history_offsets = [
        int(value)
        for value in checkpoint_value(
            args,
            checkpoint_args,
            "history_keypoint_offsets",
            [-16, -12, -8, -4],
        )
    ]
    action_frames = int(
        checkpoint_value(
            args,
            action_checkpoint_args or checkpoint_args,
            "action_target_frames",
            num_frames,
        )
    )
    action_exclude_indices = [
        int(value)
        for value in checkpoint_value(
            args,
            action_checkpoint_args or checkpoint_args,
            "action_exclude_indices",
            [],
        )
    ]
    validate_fixed_training_settings(checkpoint_args, context="Backbone checkpoint")
    validate_fixed_training_settings(action_checkpoint_args, context="Action checkpoint")
    disable_task_lang_map = bool(
        checkpoint_value(args, checkpoint_args, "disable_task_lang_map", False)
    )
    task_lang_map_path = (
        args.task_lang_map_path or checkpoint_args.get("task_lang_map_path")
    )
    if disable_task_lang_map and task_lang_map_path:
        raise ValueError(
            "--disable_task_lang_map and --task_lang_map_path are mutually exclusive"
        )
    task_lang_map = (
        load_task_lang_map(task_lang_map_path)
        if task_lang_map_path
        else not disable_task_lang_map
    )
    keypoint_dims = int(checkpoint_args.get("keypoint_dims", 3))
    use_keypoint = bool(checkpoint_args.get("use_keypoint", True))
    checkpoint_action_condition_mode = str(
        action_checkpoint_args.get(
            "action_condition_mode",
            checkpoint_args.get("action_condition_mode", "latent_keypoint"),
        )
    )
    if not use_keypoint and checkpoint_action_condition_mode != "latent_only":
        raise ValueError(
            "A w/o-keypoint checkpoint must use action_condition_mode=latent_only, "
            f"got {checkpoint_action_condition_mode!r}"
        )

    dataset_kwargs = dict(
        num_frames=num_frames,
        global_sample_stride=stride,
        action_video_freq_ratio=ratio,
        task_include=args.task_include,
        task_exclude=args.task_exclude,
        image_size=image_size,
        normalize=True,
        metadata_cache_path=args.metadata_cache_path,
        decord_num_threads=args.decord_num_threads,
        keypoint_root=args.keypoint_root,
        keypoint_dataset_key=args.keypoint_dataset_key,
        keypoint_stats_path=args.keypoint_stats_path,
        keypoint_dims=keypoint_dims,
        history_keypoint_offsets=history_offsets,
    )
    if evaluate_action:
        dataset = SkelWamVideoKeypointQposDataset(
            args.dataset_root,
            qpos_root=args.action_root,
            qpos_dataset_key=args.action_dataset_key,
            qpos_target_frames=action_frames,
            qpos_stats_path=args.action_stats_path,
            qpos_exclude_indices=action_exclude_indices,
            **dataset_kwargs,
        )
    else:
        dataset = SkelWamVideoKeypointDataset(
            args.dataset_root,
            **dataset_kwargs,
        )
    sample = dataset[int(args.sample_idx) % len(dataset)]
    video = sample["video"]
    gt_keypoints = sample["keypoints"]
    gt_valid = sample["keypoints_valid"].bool()
    history_keypoints = sample["history_keypoints"].unsqueeze(0).to(
        device=device, dtype=dtype
    )
    history_valid = sample["history_keypoints_valid"].unsqueeze(0).to(
        device=device, dtype=torch.bool
    )
    current_keypoints = gt_keypoints[0:1].unsqueeze(0).to(
        device=device, dtype=dtype
    )
    current_valid = gt_valid[0:1].unsqueeze(0).to(
        device=device, dtype=torch.bool
    )

    wan_dir = Path(args.wan_checkpoint_dir)
    transformer_dir = Path(
        args.transformer_checkpoint_dir
        or wan_dir
    )
    vae = _EVAL_RUNTIME_CACHE.get("vae")
    if vae is None:
        vae = Wan2_2_VAE(
            vae_pth=str(wan_dir / "Wan2.2_VAE.pth"),
            dtype=dtype,
            device=str(device),
        )
        _EVAL_RUNTIME_CACHE["vae"] = vae
    clean_latents = encode_video(vae, video, device=device, dtype=dtype)
    condition_latents = torch.zeros_like(clean_latents)
    condition_latents[:, :, 0:1] = clean_latents[:, :, 0:1]

    prompt_template = (
        args.prompt_template
        or checkpoint_args.get("prompt_template")
        or DEFAULT_PROMPT_TEMPLATE
    )
    empty_task_name = (
        args.empty_task_name
        or checkpoint_args.get("empty_task_name")
        or DEFAULT_EMPTY_TASK_NAME
    )
    if "language_prompts" in sample:
        prompts = [str(sample["language_prompts"])]
    else:
        prompts = build_prompts(
            str(sample["task"]),
            template=prompt_template,
            empty_task_name=empty_task_name,
            task_lang_map=task_lang_map,
        )
    precomputed_path = (
        args.precomputed_text_embeddings_path
        or checkpoint_args.get("precomputed_text_embeddings_path")
    )
    text_emb = build_text_condition(
        prompts,
        precomputed_path=precomputed_path,
        wan_checkpoint_dir=wan_dir,
        t5_checkpoint_path=args.t5_checkpoint_path,
        t5_tokenizer_path=args.t5_tokenizer_path,
        text_encoder_device=args.text_encoder_device,
        output_device=device,
        dtype=dtype,
    )

    action_dim = int(getattr(dataset, "qpos_dim", 0))
    if predicts_action:
        checkpoint_action_dim = _EVAL_RUNTIME_CACHE["checkpoint_action_dim"]
        if checkpoint_action_dim is None:
            raise KeyError(
                "Action checkpoint is missing action_fm_head.action_embedder.weight"
            )
        if evaluate_action and checkpoint_action_dim != action_dim:
            raise ValueError(
                "Action dimension mismatch between checkpoint and dataset: "
                f"{checkpoint_action_dim} vs {action_dim}"
            )
        action_dim = checkpoint_action_dim
    model = _EVAL_RUNTIME_CACHE.get("model")
    model_action_frames = (
        action_frames if predicts_action else int(dataset.video_window_size)
    )
    model_signature = (
        int(dataset.video_window_size),
        keypoint_dims,
        use_keypoint,
        model_action_frames,
        action_dim if predicts_action else 0,
        predicts_action,
        action_patch_embedding_mode,
    )
    if model is None:
        model = SkelWamBackboneModel(
            transformer_path=transformer_dir,
            keypoint_frames=int(dataset.video_window_size),
            keypoint_dims=keypoint_dims,
            keypoint_hidden_dim=int(checkpoint_args.get("keypoint_hidden_dim", 1024)),
            keypoint_ffn_dim=int(checkpoint_args.get("keypoint_ffn_dim", 4096)),
            keypoint_expert_checkpoint=None,
            keypoint_cond_video_mode=str(
                checkpoint_args.get("keypoint_cond_video_mode", "full")
            ),
            keypoint_use_text_cross_attention=bool(
                checkpoint_args.get("keypoint_use_text_cross_attention", False)
            ),
            action_frames=model_action_frames,
            action_dim=action_dim if predicts_action else 0,
            predict_action=predicts_action,
            action_condition_mode=checkpoint_action_condition_mode,
            action_patch_embedding_mode=action_patch_embedding_mode,
            use_keypoint=use_keypoint,
            torch_dtype=dtype,
            device=device,
            use_gradient_checkpointing=False,
            load_pretrained_transformer=False,
        )
        if state_dict is None:
            raise RuntimeError("Checkpoint state was released before model creation")
        load_component_state_dict(
            model,
            state_dict,
            action_state=action_state_dict,
        )
        model.eval().requires_grad_(False)
        _EVAL_RUNTIME_CACHE["model"] = model
        _EVAL_RUNTIME_CACHE["model_signature"] = model_signature
        _EVAL_RUNTIME_CACHE.pop("state_dict", None)
        _EVAL_RUNTIME_CACHE.pop("action_state_dict", None)
    elif _EVAL_RUNTIME_CACHE.get("model_signature") != model_signature:
        raise ValueError(
            "The cached checkpoint model is incompatible with the next dataset: "
            f"cached={_EVAL_RUNTIME_CACHE.get('model_signature')}, "
            f"requested={model_signature}."
        )

    latent_steps = args.latent_num_inference_steps or args.num_inference_steps
    keypoint_steps = args.keypoint_num_inference_steps or args.num_inference_steps
    action_steps = args.action_num_inference_steps or args.num_inference_steps
    latent_shift = float(
        args.latent_snr_shift
        if args.latent_snr_shift is not None
        else checkpoint_args.get("latent_snr_shift", 5.0)
    )
    keypoint_shift = float(
        args.keypoint_snr_shift
        if args.keypoint_snr_shift is not None
        else checkpoint_args.get("keypoint_snr_shift", 1.0)
    )
    action_shift = float(
        args.action_snr_shift
        if args.action_snr_shift is not None
        else checkpoint_args.get("action_snr_shift", 1.0)
    )
    latent_scheduler = FlowMatchingScheduler(
        num_inference_steps=latent_steps, snr_shift=latent_shift
    )
    latent_scheduler.set_timesteps(latent_steps, device=device, dtype=dtype)
    pred_latents = sample_video_latents(
        model,
        latent_scheduler,
        condition_latents=condition_latents,
        history_keypoints=history_keypoints,
        current_keypoints=current_keypoints,
        text_emb=text_emb,
        history_keypoints_valid=history_valid,
        current_keypoints_valid=current_valid,
        seed=args.seed,
    )

    pred_keypoint_state = None
    pred_keypoints_raw = None
    pred_valid = None
    pred_keypoints = None
    kn_vc_attention_stages = None
    if use_keypoint:
        keypoint_condition_latents = (
            pred_latents if args.keypoint_video_condition == "pred" else clean_latents
        )
        keypoint_scheduler = FlowMatchingScheduler(
            num_inference_steps=keypoint_steps, snr_shift=keypoint_shift
        )
        keypoint_scheduler.set_timesteps(keypoint_steps, device=device, dtype=dtype)
        pred_keypoint_state, kn_vc_attention_stages = sample_keypoint_state(
            model,
            keypoint_scheduler,
            condition_latents=keypoint_condition_latents,
            current_keypoints=current_keypoints,
            current_keypoints_valid=current_valid,
            text_emb=text_emb,
            seed=args.seed + 1,
            capture_attention=(
                args.vis_keypoint_video_attn
                or args.vis_keypoint_video_same_frame_attn
            ),
        )
        pred_keypoints_raw = pred_keypoint_state[..., :keypoint_dims]
        pred_valid = decode_keypoint_validity(
            pred_keypoint_state[..., keypoint_dims],
        )
        pred_valid[:, 0:1] = current_valid
        pred_keypoints = torch.where(
            pred_valid.unsqueeze(-1),
            pred_keypoints_raw,
            pred_keypoints_raw.new_zeros(()),
        )
    elif args.vis_keypoint_video_attn or args.vis_keypoint_video_same_frame_attn:
        raise ValueError(
            "Keypoint attention visualization is unavailable for a w/o-keypoint checkpoint"
        )

    pred_action = None
    action_modality_contribution_stages = None
    if args.vis_action_modality_contribution and not evaluate_action:
        raise ValueError(
            "--vis_action_modality_contribution requires --run_action and an "
            "Action-capable checkpoint"
        )
    if (
        args.vis_action_modality_contribution
        and model.action_condition_mode != "latent_keypoint"
    ):
        raise ValueError(
            "--vis_action_modality_contribution requires "
            "action_condition_mode=latent_keypoint"
        )
    if evaluate_action:
        if args.action_condition == "gt":
            action_latents = clean_latents
            action_keypoints = gt_keypoints.unsqueeze(0).to(device=device, dtype=dtype)
            action_keypoints_valid = gt_valid.unsqueeze(0).to(device=device)
        else:
            action_latents = pred_latents
            action_keypoints = pred_keypoints
            action_keypoints_valid = pred_valid
        if model.action_condition_mode == "latent_only":
            action_keypoints = None
            action_keypoints_valid = None
        action_scheduler = FlowMatchingScheduler(
            num_inference_steps=action_steps, snr_shift=action_shift
        )
        action_scheduler.set_timesteps(action_steps, device=device, dtype=dtype)
        action_sample = sample_action(
            model,
            action_scheduler,
            condition_latents=action_latents,
            condition_keypoints=action_keypoints,
            condition_keypoints_valid=action_keypoints_valid,
            seed=args.seed + 2,
            capture_modality_contributions=(
                args.vis_action_modality_contribution
            ),
        )
        if args.vis_action_modality_contribution:
            if not isinstance(action_sample, tuple):
                raise RuntimeError("Action modality contributions were not captured")
            pred_action, action_modality_contribution_stages = action_sample
        else:
            if isinstance(action_sample, tuple):
                raise RuntimeError("Unexpected Action modality contribution capture")
            pred_action = action_sample

    pred_video = decode_video(vae, pred_latents)
    reconstructed_video = decode_video(vae, clean_latents)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_video(
        pred_video.unsqueeze(0),
        output_dir / "pred_video.mp4",
        fps=args.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    save_video(
        video.unsqueeze(0),
        output_dir / "gt_video.mp4",
        fps=args.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    save_video(
        reconstructed_video.unsqueeze(0),
        output_dir / "vae_reconstruction.mp4",
        fps=args.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    gt_unit = video_to_unit(video)[:, 1:]
    pred_unit = video_to_unit(pred_video)[:, 1:]
    reconstructed_unit = video_to_unit(reconstructed_video)[:, 1:]
    metrics = {
        "video_psnr_rollout_gt_future": video_psnr(pred_unit, gt_unit),
        "video_ssim_rollout_gt_future": video_ssim(pred_unit, gt_unit),
        "video_psnr_rollout_decode_future": video_psnr(
            pred_unit, reconstructed_unit
        ),
        "video_ssim_rollout_decode_future": video_ssim(
            pred_unit, reconstructed_unit
        ),
        "video_psnr_decode_gt_future": video_psnr(reconstructed_unit, gt_unit),
        "video_ssim_decode_gt_future": video_ssim(reconstructed_unit, gt_unit),
    }
    if use_keypoint:
        valid_future = gt_valid[1:].to(device=device)
        metrics.update(
            classification_metrics(pred_valid[0, 1:], valid_future)
        )
        xyz_error = (
            pred_keypoints_raw[0, 1:].float()
            - gt_keypoints[1:].to(device=device).float()
        )
        xyz_mask = valid_future.unsqueeze(-1).expand_as(xyz_error)
        if xyz_mask.any():
            metrics["keypoint_xyz_mae_normalized"] = float(
                xyz_error.abs()[xyz_mask].mean().item()
            )
            metrics["keypoint_xyz_rmse_normalized"] = float(
                xyz_error.square()[xyz_mask].mean().sqrt().item()
            )

    gt_keypoints_vis = denormalize_keypoints(sample, gt_keypoints)
    pred_frames = video_to_uint8(pred_video)
    gt_frames = video_to_uint8(video)
    reconstructed_frames = video_to_uint8(reconstructed_video)
    gt_overlay = overlay_keypoints(
        gt_frames,
        gt_keypoints_vis,
        gt_valid,
        color=(50, 220, 70),
        radius=args.overlay_radius,
    )
    save_frames(output_dir / "gt_keypoint_overlay.mp4", gt_overlay, fps=args.fps)
    pred_overlay = None
    if use_keypoint:
        pred_keypoints_vis = denormalize_keypoints(
            sample, pred_keypoints[0].detach().cpu()
        )
        pred_overlay = overlay_keypoints(
            pred_frames,
            pred_keypoints_vis,
            pred_valid[0].cpu(),
            color=(245, 70, 70),
            radius=args.overlay_radius,
        )
        save_frames(
            output_dir / "pred_keypoint_overlay.mp4", pred_overlay, fps=args.fps
        )
    save_frames(
        output_dir / "video_triptych.mp4",
        [
            np.concatenate(frames, axis=1)
            for frames in zip(
                gt_frames,
                reconstructed_frames,
                pred_frames,
            )
        ],
        fps=args.fps,
    )
    attention_outputs: dict[str, Any] = {}
    if args.vis_keypoint_video_attn:
        if not kn_vc_attention_stages:
            raise RuntimeError(
                "--vis_keypoint_video_attn was enabled but no attention was captured"
            )
        attention_condition_frames = (
            pred_frames
            if args.keypoint_video_condition == "pred"
            else reconstructed_frames
        )
        attention_outputs = save_kn_vc_attention_visualizations(
            kn_vc_attention_stages,
            condition_frames=attention_condition_frames,
            num_hands=model.keypoint_expert.num_hands,
            output_dir=output_dir,
            fps=args.fps,
            keep_height_ratio=args.attn_vis_keep_height_ratio,
        )
    if args.vis_keypoint_video_same_frame_attn:
        if not kn_vc_attention_stages:
            raise RuntimeError(
                "--vis_keypoint_video_same_frame_attn was enabled but no "
                "attention was captured"
            )
        attention_condition_frames = (
            pred_frames
            if args.keypoint_video_condition == "pred"
            else reconstructed_frames
        )
        attention_outputs.update(
            save_kn_vc_same_frame_attention_visualizations(
                kn_vc_attention_stages,
                condition_frames=attention_condition_frames,
                num_hands=model.keypoint_expert.num_hands,
                output_dir=output_dir,
                fps=args.fps,
                keep_height_ratio=args.attn_vis_keep_height_ratio,
            )
        )
    if args.vis_action_modality_contribution:
        if not action_modality_contribution_stages:
            raise RuntimeError(
                "--vis_action_modality_contribution was enabled but no "
                "contributions were captured"
            )
        attention_outputs.update(
            save_action_modality_contribution_visualizations(
                action_modality_contribution_stages,
                output_dir=output_dir,
            )
        )
    if pred_overlay is not None:
        save_frames(
            output_dir / "keypoint_overlay_side_by_side.mp4",
            [
                np.concatenate(frames, axis=1)
                for frames in zip(gt_overlay, pred_overlay)
            ],
            fps=args.fps,
        )

    gt_action = (
        sample["qpos"].to(device=device, dtype=dtype)
        if "qpos" in sample
        else None
    )
    action_valid = (
        torch.isfinite(gt_action).all(dim=-1)
        if gt_action is not None
        else None
    )
    if (
        pred_action is not None
        and gt_action is not None
        and action_valid is not None
        and action_valid.any()
    ):
        action_error = pred_action[0].float() - gt_action.float()
        metrics["action_mae_normalized"] = float(
            action_error.abs()[action_valid].mean().item()
        )
        metrics["action_rmse_normalized"] = float(
            action_error.square()[action_valid].mean().sqrt().item()
        )
        pred_action_denormalized = denormalize_action(
            sample, pred_action[0].detach().float().cpu()
        )
        gt_action_denormalized = denormalize_action(
            sample, sample["qpos"].detach().float().cpu()
        )
        if dataset.raw_qpos_dim == 42:
            if tuple(dataset.qpos_indices) != tuple(range(42)):
                raise ValueError("Real-robot physical metrics require all 42 action dimensions in order.")
            metrics.update(real_robot_action_metrics(
                pred_action_denormalized, gt_action_denormalized, action_valid.cpu()
            ))
        else:
            denormalized_error = pred_action_denormalized - gt_action_denormalized
            metrics["action_mae(rad)"] = float(
                denormalized_error.abs()[action_valid.cpu()].mean().item()
            )
            metrics["action_mae(deg)"] = metrics["action_mae(rad)"] / np.pi * 180
            metrics["action_rmse(rad)"] = float(
                denormalized_error.square()[action_valid.cpu()].mean().sqrt().item()
            )
            metrics["action_rmse(deg)"] = metrics["action_rmse(rad)"] / np.pi * 180
        plot_action(
            pred_action_denormalized,
            gt_action_denormalized,
            action_valid.cpu(),
            output_dir / "action_plot.png",
        )

    predictions = {
        "pred_keypoints_normalized": tensor_json(
            pred_keypoints[0] if pred_keypoints is not None else None
        ),
        "pred_keypoints_raw_normalized": tensor_json(
            pred_keypoints_raw[0] if pred_keypoints_raw is not None else None
        ),
        "pred_keypoints_valid": tensor_json(
            pred_valid[0] if pred_valid is not None else None
        ),
        "gt_keypoints_normalized": tensor_json(gt_keypoints),
        "gt_keypoints_valid": tensor_json(gt_valid),
        "pred_action_normalized": tensor_json(
            pred_action[0] if pred_action is not None else None
        ),
        "gt_action_normalized": tensor_json(gt_action),
    }
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "use_keypoint": use_keypoint,
        "action_condition_mode": checkpoint_action_condition_mode,
        "dataset_type": args.dataset_type,
        "sample_idx": int(args.sample_idx),
        "task": str(sample["task"]),
        "episode_path": str(sample["episode_path"]),
        "prompt": prompts[0],
        "video_frames": int(video.shape[1]),
        "action_frames": action_frames if evaluate_action else None,
        "action_dim": action_dim,
        "latent_num_inference_steps": latent_steps,
        "keypoint_num_inference_steps": keypoint_steps if use_keypoint else None,
        "action_num_inference_steps": action_steps if pred_action is not None else None,
        "latent_snr_shift": latent_shift,
        "keypoint_snr_shift": keypoint_shift if use_keypoint else None,
        "action_snr_shift": action_shift if pred_action is not None else None,
        "keypoint_video_condition": (
            args.keypoint_video_condition if use_keypoint else None
        ),
        "action_condition": args.action_condition if pred_action is not None else None,
        "vis_keypoint_video_attn": bool(args.vis_keypoint_video_attn),
        "vis_keypoint_video_same_frame_attn": bool(
            args.vis_keypoint_video_same_frame_attn
        ),
        "vis_action_modality_contribution": bool(
            args.vis_action_modality_contribution
        ),
        "attention_outputs": attention_outputs,
        "metrics": metrics,
    }
    if "action_rotation_valid_count" in metrics:
        summary["action_metrics_metadata"] = {
            "layout": "right_21_then_left_21",
            "rotation": "geodesic angle after Gram-Schmidt decoding of two rotation columns",
            "translation_unit": "original arm_ee_pose translation unit (no conversion)",
            "translation_mae_rmse": "pooled xyz coordinate errors",
            "translation_l2_mean": "mean 3D distance per hand and frame",
            "hand_joint_unit": "radians, also reported in degrees",
            "aggregation": "both hands and all finite-target frames, including boundary copies",
            "degenerate_rotations": "excluded from rotation errors and counted separately; null if none valid",
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
