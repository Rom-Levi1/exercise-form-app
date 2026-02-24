from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any


@dataclass
class PosePoint:
    x: float
    y: float
    z: float
    visibility: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PoseFrame:
    frameIndex: int
    timestampSec: float
    hasPose: bool
    landmarks: Dict[str, PosePoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frameIndex": self.frameIndex,
            "timestampSec": self.timestampSec,
            "hasPose": self.hasPose,
            "landmarks": {name: point.to_dict() for name, point in self.landmarks.items()},
        }


# MediaPipe Pose landmark names in index order (33 landmarks)
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]