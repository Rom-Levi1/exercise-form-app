import cv2
from typing import Generator, Tuple, Dict, Any


def get_video_metadata(videoPath: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {videoPath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    durationSec = (frameCount / fps) if fps > 0 else 0.0

    cap.release()

    return {
        "fps": fps,
        "frameCount": frameCount,
        "width": width,
        "height": height,
        "durationSec": durationSec,
    }


def iter_video_frames(videoPath: str) -> Generator[Tuple[int, float, any], None, None]:
    """
    Yields: (frameIndex, timestampSec, frameBgr)
    """
    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {videoPath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frameIndex = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestampSec = (frameIndex / fps) if fps > 0 else 0.0
            yield frameIndex, timestampSec, frame
            frameIndex += 1
    finally:
        cap.release()