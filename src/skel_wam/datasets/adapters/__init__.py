"""Adapters for video and video-keypoint source datasets."""
from .egovla_general import VideoKeypointSourceAdapter, EgoVlaGeneralDataset
from .real_video_general import VideoSourceAdapter, RealVideoGeneralDataset

__all__ = ["VideoKeypointSourceAdapter", "VideoSourceAdapter", "EgoVlaGeneralDataset", "RealVideoGeneralDataset"]
