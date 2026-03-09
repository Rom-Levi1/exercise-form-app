from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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

FACE_LANDMARK_NAMES = {
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
}


def _norm_to_px(normX: float, normY: float, width: int, height: int) -> Tuple[int, int]:
    x = int(max(0.0, min(1.0, normX)) * (width - 1))
    y = int(max(0.0, min(1.0, normY)) * (height - 1))
    return x, y


def _get_frame_events(analysisResult: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        analysisResult.get("metrics", {})
        .get("visualFeedback", {})
        .get("events", [])
    )

def _issue_code_to_message(issueCode: str) -> str:
    mapping = {
        "depth_high": "Go deeper at the bottom.",
        "depth_moderate": "Slightly deeper squat depth.",
        "torso_lean_excessive": "Keep torso more upright.",
        "torso_lean_moderate": "Try to reduce forward lean.",
        "lockout_incomplete": "Finish with full knee lockout.",
    }
    return mapping.get(issueCode, issueCode.replace("_", " "))


def _wrap_text(text: str, font, fontScale: float, thickness: int, maxWidthPx: int) -> List[str]:
    words = text.split()
    if not words:
        return [text]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidateWidth = cv2.getTextSize(candidate, font, fontScale, thickness)[0][0]
        if candidateWidth <= maxWidthPx:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _build_rep_summaries(analysisResult: Dict[str, Any]) -> List[Dict[str, Any]]:
    repFeedback = analysisResult.get("repFeedback", [])
    if not isinstance(repFeedback, list):
        return []

    events = _get_frame_events(analysisResult)
    eventsByRep: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        rep = event.get("rep")
        if isinstance(rep, int):
            eventsByRep.setdefault(rep, []).append(event)

    summaries: List[Dict[str, Any]] = []
    for repData in repFeedback:
        rep = repData.get("rep")
        startFrame = repData.get("startFrameIndex")
        endFrame = repData.get("endFrameIndex")
        quality = repData.get("quality")
        issues = repData.get("issues", [])

        if not isinstance(rep, int) or not isinstance(startFrame, int) or not isinstance(endFrame, int):
            continue
        if not isinstance(issues, list):
            issues = []

        repEvents = eventsByRep.get(rep, [])
        eventMessages: List[str] = []
        for event in repEvents:
            message = event.get("message")
            if isinstance(message, str) and message and message not in eventMessages:
                eventMessages.append(message)

        issueMessages = eventMessages or [_issue_code_to_message(str(code)) for code in issues]
        summaries.append(
            {
                "rep": rep,
                "startFrameIndex": startFrame,
                "endFrameIndex": endFrame,
                "quality": quality,
                "issues": issues,
                "issueMessages": issueMessages,
            }
        )

    summaries.sort(key=lambda item: item["rep"])
    return summaries


def _current_rep_number(frameIndex: int, repSummaries: List[Dict[str, Any]]) -> int:
    if not repSummaries:
        return 0

    for summary in repSummaries:
        if summary["startFrameIndex"] <= frameIndex <= summary["endFrameIndex"]:
            return summary["rep"]

    # Before first rep starts
    firstStart = repSummaries[0]["startFrameIndex"]
    if frameIndex < firstStart:
        return 0

    # After a rep ended but before next starts, keep latest completed rep index.
    completed = [s["rep"] for s in repSummaries if frameIndex >= s["endFrameIndex"]]
    if completed:
        return max(completed)

    return 0


def _draw_side_panel(
    frameBgr,
    panelX: int,
    panelWidth: int,
    repSummaries: List[Dict[str, Any]],
    currentRep: int,
    pauseSummary: Optional[Dict[str, Any]],
) -> None:
    height, width = frameBgr.shape[:2]
    cv2.rectangle(frameBgr, (panelX, 0), (width, height), (245, 245, 245), -1)
    cv2.line(frameBgr, (panelX, 0), (panelX, height), (210, 210, 210), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    left = panelX + 22
    y = 42
    maxTextWidth = panelWidth - 44

    cv2.putText(frameBgr, "Squats", (left, y), font, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    y += 36
    cv2.line(frameBgr, (left, y), (panelX + panelWidth - 22, y), (210, 210, 210), 1)
    y += 34

    totalReps = len(repSummaries)
    repLabel = f"Rep {max(0, currentRep)} / {totalReps}"
    cv2.putText(frameBgr, repLabel, (left, y), font, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
    y += 34

    if pauseSummary is None:
        status = "Tracking..."
        helper = "Feedback appears after each rep."
        cv2.putText(frameBgr, status, (left, y), font, 0.62, (70, 70, 70), 2, cv2.LINE_AA)
        y += 32
        for line in _wrap_text(helper, font, 0.55, 1, maxTextWidth):
            cv2.putText(frameBgr, line, (left, y), font, 0.55, (95, 95, 95), 1, cv2.LINE_AA)
            y += 24
        return

    issues = pauseSummary.get("issues", [])
    quality = pauseSummary.get("quality")
    isGoodRep = isinstance(issues, list) and len(issues) == 0

    if isGoodRep:
        title = "Good rep."
        subtitle = "Depth, torso angle, and lockout looked good."
        color = (40, 140, 40)
        cv2.putText(frameBgr, title, (left, y), font, 0.72, color, 2, cv2.LINE_AA)
        y += 34
        for line in _wrap_text(subtitle, font, 0.57, 1, maxTextWidth):
            cv2.putText(frameBgr, line, (left, y), font, 0.57, (45, 45, 45), 1, cv2.LINE_AA)
            y += 24
    else:
        title = "Needs work:"
        cv2.putText(frameBgr, title, (left, y), font, 0.72, (30, 30, 180), 2, cv2.LINE_AA)
        y += 34
        messages = pauseSummary.get("issueMessages", [])
        if not isinstance(messages, list):
            messages = []

        for message in messages[:3]:
            for idx, line in enumerate(_wrap_text(f"- {message}", font, 0.57, 1, maxTextWidth)):
                cv2.putText(frameBgr, line, (left, y), font, 0.57, (45, 45, 45), 1, cv2.LINE_AA)
                y += 24
                if idx > 2:
                    break

    if isinstance(quality, (int, float)):
        y += 12
        qualityInt = int(round(float(quality)))
        cv2.putText(
            frameBgr,
            f"Rep score: {qualityInt}/100",
            (left, y),
            font,
            0.60,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
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

    for name, point in landmarks.items():
        if name in FACE_LANDMARK_NAMES:
            continue
        if point.visibility < minVisibility:
            continue
        center = _norm_to_px(point.x, point.y, width, height)
        cv2.circle(frameBgr, center, 4, (80, 255, 80), -1, cv2.LINE_AA)


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
    - pauses after every completed rep
    - renders text in a fixed side panel for consistent readability
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

    repSummaries = _build_rep_summaries(analysisResult)
    repByEndFrame: Dict[int, Dict[str, Any]] = {
        summary["endFrameIndex"]: summary for summary in repSummaries
    }

    pauseFrames = max(0, int(round(fps * pauseSeconds)))
    panelWidth = max(280, int(round(width * 0.38)))
    outputWidth = width + panelWidth

    outputP = Path(outputPath)
    outputP.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(outputP),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (outputWidth, height),
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

            pauseSummary = repByEndFrame.get(frameIndex)
            currentRep = _current_rep_number(frameIndex, repSummaries)

            composedFrame = np.full((height, outputWidth, 3), 245, dtype=np.uint8)
            composedFrame[:, :width] = frameOut
            _draw_side_panel(
                composedFrame,
                panelX=width,
                panelWidth=panelWidth,
                repSummaries=repSummaries,
                currentRep=currentRep,
                pauseSummary=pauseSummary,
            )

            writer.write(composedFrame)

            if pauseSummary is not None and pauseFrames > 0:
                for _ in range(pauseFrames):
                    writer.write(composedFrame)
    finally:
        writer.release()

    return str(outputP)
