import cv2
import mediapipe as mp
from typing import List, Dict

from backend.core.video.video_reader import iter_video_frames, get_video_metadata
from backend.core.pose.landmark_schema import PosePoint, PoseFrame, LANDMARK_NAMES


mpPose = mp.solutions.pose


def _landmarks_to_dict(poseLandmarks) -> Dict[str, PosePoint]:
    landmarksDict: Dict[str, PosePoint] = {}

    if poseLandmarks is None:
        return landmarksDict

    for index, landmark in enumerate(poseLandmarks.landmark):
        if index >= len(LANDMARK_NAMES):
            break

        name = LANDMARK_NAMES[index]
        landmarksDict[name] = PosePoint(
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(landmark.z),
            visibility=float(landmark.visibility),
        )

    return landmarksDict


def extract_pose_frames(
    videoPath: str,
    modelComplexity: int = 1,
    minDetectionConfidence: float = 0.5,
    minTrackingConfidence: float = 0.5,
) -> Dict[str, object]:
    """
    Returns a dict with:
      - videoMetadata
      - poseFrames (list[PoseFrame])
    """
    videoMetadata = get_video_metadata(videoPath)
    poseFrames: List[PoseFrame] = []

    with mpPose.Pose(
        static_image_mode=False,
        model_complexity=modelComplexity,
        smooth_landmarks=True,
        min_detection_confidence=minDetectionConfidence,
        min_tracking_confidence=minTrackingConfidence,
    ) as pose:
        for frameIndex, timestampSec, frameBgr in iter_video_frames(videoPath):
            frameRgb = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRgb)

            landmarksDict = _landmarks_to_dict(results.pose_landmarks)
            hasPose = len(landmarksDict) > 0

            poseFrames.append(
                PoseFrame(
                    frameIndex=frameIndex,
                    timestampSec=timestampSec,
                    hasPose=hasPose,
                    landmarks=landmarksDict,
                )
            )

    return {
        "videoMetadata": videoMetadata,
        "poseFrames": poseFrames,
    }