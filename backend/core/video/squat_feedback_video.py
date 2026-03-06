from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from backend.core.video.video_reader import get_video_metadata, iter_video_frames


# MediaPipe pose landmark pairs by name for a clear, lightweight skeleton.
POSE_CONNECTIONS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]


def _norm_to_px(normX: float, normY: float, width: int, height: int) -> Tuple[int, int]:
    x = int(max(0.0, min(1.0, normX)) * (width - 1))
    y = int(max(0.0, min(1.0, normY)) * (height - 1))
    return x, y


def _pick_color(severity: str) -> Tuple[int, int, int]:
    if severity == "high":
        return (0, 0, 255)  # red
    if severity == "medium":
        return (0, 165, 255)  # orange
    return (0, 255, 255)  # yellow


def _get_frame_events(analysisResult: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        analysisResult.get("metrics", {})
        .get("visualFeedback", {})
        .get("events", [])
    )


def _draw_skeleton(frameBgr, landmarks: Dict[str, Any], minVisibility: float = 0.4) -> None:
    height, width = frameBgr.shape[:2]

    for firstName, secondName in POSE_CONNECTIONS:
        first = landmarks.get(firstName)
        second = landmarks.get(secondName)
        if first is None or second is None:
            continue
        if first.visibility < minVisibility or second.visibility < minVisibility:
            continue

        p1 = _norm_to_px(first.x, first.y, width, height)
        p2 = _norm_to_px(second.x, second.y, width, height)
        cv2.line(frameBgr, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)

    for point in landmarks.values():
        if point.visibility < minVisibility:
            continue
        center = _norm_to_px(point.x, point.y, width, height)
        cv2.circle(frameBgr, center, 4, (80, 255, 80), -1, cv2.LINE_AA)


def _draw_feedback_banner(frameBgr, events: List[Dict[str, Any]]) -> None:
    if not events:
        return

    height, width = frameBgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Keep the original readable look and only wrap to avoid horizontal cutoff.
    fontScale = 0.66
    thickness = 2
    lineHeight = 34
    leftPad = 18
    rightPad = 18
    topPad = 14
    bottomPad = 12
    maxTextWidth = max(120, width - leftPad - rightPad)

    def wrap_text(text: str) -> List[str]:
        words = text.split()
        if not words:
            return [text]

        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            candidateWidth = cv2.getTextSize(candidate, font, fontScale, thickness)[0][0]
            if candidateWidth <= maxTextWidth:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    # Build wrapped lines with per-line color.
    renderedLines: List[Tuple[str, Tuple[int, int, int]]] = []
    for idx, event in enumerate(events):
        if idx >= 3:
            break

        severity = str(event.get("severity", "low"))
        color = _pick_color(severity)
        msg = str(event.get("message", "Form issue detected"))
        measured = event.get("measuredAngleDeg")
        target = event.get("targetAngleDeg")

        if measured is not None and target is not None:
            text = f"{msg} ({measured:.1f} deg, target {target:.1f} deg)"
        else:
            text = msg

        for line in wrap_text(text):
            renderedLines.append((line, color))

    # Keep banner readable without covering too much of the frame.
    maxLines = max(3, int((height * 0.4 - topPad - bottomPad) // lineHeight))
    if len(renderedLines) > maxLines:
        renderedLines = renderedLines[:maxLines]
        lastText, lastColor = renderedLines[-1]
        renderedLines[-1] = (f"{lastText} ...", lastColor)

    bannerHeight = topPad + bottomPad + lineHeight * len(renderedLines)
    overlay = frameBgr.copy()
    cv2.rectangle(overlay, (0, 0), (width, bannerHeight), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frameBgr, 0.28, 0, frameBgr)

    lineY = topPad + lineHeight - 4
    for text, color in renderedLines:
        cv2.putText(
            frameBgr,
            text,
            (leftPad, lineY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        lineY += lineHeight


def create_squat_feedback_video(
    videoPath: str,
    poseFrames: List[Any],
    analysisResult: Dict[str, Any],
    outputPath: str,
    pauseSeconds: float = 2.0,
    minVisibility: float = 0.4,
) -> Optional[str]:
    """
    Generates an annotated squat feedback video:
    - draws pose skeleton on every frame
    - pauses on error event frames and overlays the error text + angle details
    Returns the output path if written, else None.
    """
    if not poseFrames:
        return None

    metadata = get_video_metadata(videoPath)
    fps = float(metadata.get("fps") or 0.0)
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)

    if fps <= 0 or width <= 0 or height <= 0:
        return None

    allEvents = _get_frame_events(analysisResult)
    if not isinstance(allEvents, list):
        allEvents = []

    eventsByFrame: Dict[int, List[Dict[str, Any]]] = {}
    for event in allEvents:
        frameIndex = event.get("frameIndex")
        if not isinstance(frameIndex, int):
            continue
        eventsByFrame.setdefault(frameIndex, []).append(event)

    pauseFrames = max(0, int(round(fps * pauseSeconds)))

    outputP = Path(outputPath)
    outputP.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(outputP),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        return None

    try:
        for frameIndex, _, frameBgr in iter_video_frames(videoPath):
            frameOut = frameBgr.copy()

            if frameIndex < len(poseFrames):
                poseFrame = poseFrames[frameIndex]
                if getattr(poseFrame, "hasPose", False):
                    _draw_skeleton(frameOut, poseFrame.landmarks, minVisibility=minVisibility)

            frameEvents = eventsByFrame.get(frameIndex, [])
            if frameEvents:
                _draw_feedback_banner(frameOut, frameEvents)

            writer.write(frameOut)

            if frameEvents and pauseFrames > 0:
                for _ in range(pauseFrames):
                    writer.write(frameOut)
    finally:
        writer.release()

    return str(outputP)
