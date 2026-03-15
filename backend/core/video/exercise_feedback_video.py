from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _humanize_check_name(name: str) -> str:
    text = name.replace("_", " ")
    if text.endswith(" ok"):
        text = text[:-3]
    return text.capitalize()


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


def build_rep_summaries_from_analysis_result(
    analysisResult: Dict[str, Any],
    issueMessageResolver: Optional[Callable[[str], str]] = None,
    positiveStatus: str = "Good rep.",
    positiveDetailLines: Optional[List[str]] = None,
    repKeyCandidates: Tuple[str, ...] = ("rep", "repIndex"),
) -> List[Dict[str, Any]]:
    repFeedback = analysisResult.get("repFeedback", [])
    if not isinstance(repFeedback, list):
        return []

    positiveLines = positiveDetailLines or ["Form looked good in this rep."]
    events = _get_frame_events(analysisResult)
    eventsByRep: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        rep = event.get("rep")
        if isinstance(rep, int):
            eventsByRep.setdefault(rep, []).append(event)

    summaries: List[Dict[str, Any]] = []
    for repData in repFeedback:
        repNumber = None
        for key in repKeyCandidates:
            candidate = repData.get(key)
            if isinstance(candidate, int):
                repNumber = candidate
                break

        startFrame = repData.get("startFrameIndex")
        endFrame = repData.get("endFrameIndex")
        pauseFrame = repData.get("pauseFrameIndex")
        if not isinstance(pauseFrame, int):
            pauseFrame = repData.get("topFrameIndex")
        if not isinstance(pauseFrame, int):
            pauseFrame = endFrame
        quality = repData.get("quality")
        issues = repData.get("issues", [])

        if not isinstance(repNumber, int):
            continue
        if not isinstance(startFrame, int) or not isinstance(endFrame, int):
            continue
        if not isinstance(issues, list):
            issues = []

        repEvents = eventsByRep.get(repNumber, [])
        detailLines: List[str] = []
        for event in repEvents:
            message = event.get("message")
            if isinstance(message, str) and message and message not in detailLines:
                detailLines.append(message)

        if not detailLines:
            for issueCode in issues:
                issueText = (
                    issueMessageResolver(str(issueCode))
                    if issueMessageResolver is not None
                    else str(issueCode).replace("_", " ")
                )
                if issueText not in detailLines:
                    detailLines.append(issueText)

        isGoodRep = len(issues) == 0
        summaries.append(
            {
                "rep": repNumber,
                "startFrameIndex": startFrame,
                "endFrameIndex": endFrame,
                "pauseFrameIndex": pauseFrame,
                "quality": quality,
                "issues": issues,
                "status": positiveStatus if isGoodRep else "Needs work:",
                "detailLines": positiveLines if isGoodRep else detailLines[:3],
                "isGoodRep": isGoodRep,
            }
        )

    summaries.sort(key=lambda item: item["rep"])
    return summaries


def build_rep_summaries_from_text_feedback(
    analysisResult: Dict[str, Any],
    textFeedback: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    repFeedback = analysisResult.get("repFeedback", [])
    if not isinstance(repFeedback, list):
        return []

    repBreakdown = (textFeedback or {}).get("repBreakdown", [])
    if not isinstance(repBreakdown, list):
        return []

    repFeedbackByNumber: Dict[int, Dict[str, Any]] = {}
    for repData in repFeedback:
        repNumber = repData.get("rep")
        if not isinstance(repNumber, int):
            repNumber = repData.get("repIndex")
        if isinstance(repNumber, int):
            repFeedbackByNumber[repNumber] = repData

    summaries: List[Dict[str, Any]] = []
    for breakdownItem in repBreakdown:
        if not isinstance(breakdownItem, dict):
            continue

        repNumber = breakdownItem.get("rep")
        if not isinstance(repNumber, int):
            continue

        repData = repFeedbackByNumber.get(repNumber)
        if not isinstance(repData, dict):
            continue

        startFrame = repData.get("startFrameIndex")
        endFrame = repData.get("endFrameIndex")
        pauseFrame = repData.get("pauseFrameIndex")
        if not isinstance(pauseFrame, int):
            pauseFrame = repData.get("topFrameIndex")
        if not isinstance(pauseFrame, int):
            pauseFrame = endFrame

        if not isinstance(startFrame, int) or not isinstance(endFrame, int):
            continue

        detailsText = breakdownItem.get("details")
        detailLines: List[str] = []
        if isinstance(detailsText, str) and detailsText.strip():
            detailLines = [detailsText.strip()]

        issues = repData.get("issues", [])
        if not isinstance(issues, list):
            issues = []

        summaries.append(
            {
                "rep": repNumber,
                "startFrameIndex": startFrame,
                "endFrameIndex": endFrame,
                "pauseFrameIndex": pauseFrame,
                "quality": repData.get("quality"),
                "issues": issues,
                "headline": breakdownItem.get("label") or ("Strong rep" if not issues else "Needs adjustment"),
                "detailLines": detailLines,
                "rating": breakdownItem.get("rating"),
                "isGoodRep": len(issues) == 0,
            }
        )

    summaries.sort(key=lambda item: item["rep"])
    return summaries


def build_rep_summaries_from_boolean_checks(
    repFeedback: List[Dict[str, Any]],
    repWindows: List[Dict[str, int]],
    checkLabels: Dict[str, str],
    positiveStatus: str = "Good rep.",
    positiveDetailLines: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    positiveLines = positiveDetailLines or ["Form looked good in this rep."]
    summaries: List[Dict[str, Any]] = []

    for repData, repWindow in zip(repFeedback, repWindows):
        repNumber = repData.get("rep")
        if not isinstance(repNumber, int):
            repNumber = repData.get("repIndex")
        if not isinstance(repNumber, int):
            continue

        failedChecks: List[str] = []
        for key, label in checkLabels.items():
            value = repData.get(key)
            if isinstance(value, bool) and not value:
                failedChecks.append(label)

        isGoodRep = len(failedChecks) == 0
        summaries.append(
            {
                "rep": repNumber,
                "startFrameIndex": repWindow["startFrameIndex"],
                "endFrameIndex": repWindow["endFrameIndex"],
                "quality": repData.get("quality"),
                "issues": failedChecks,
                "status": positiveStatus if isGoodRep else "Needs work:",
                "detailLines": positiveLines if isGoodRep else failedChecks[:3],
                "isGoodRep": isGoodRep,
            }
        )

    summaries.sort(key=lambda item: item["rep"])
    return summaries


def infer_detail_lines_from_boolean_checks(
    repData: Dict[str, Any],
    preferredKeys: Optional[List[str]] = None,
    labelOverrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    keysToCheck = preferredKeys or [key for key in repData.keys() if key.endswith("Ok")]
    labels = labelOverrides or {}
    lines: List[str] = []

    for key in keysToCheck:
        value = repData.get(key)
        if isinstance(value, bool) and not value:
            lines.append(labels.get(key, _humanize_check_name(key)))

    return lines


def _current_rep_number(frameIndex: int, repSummaries: List[Dict[str, Any]]) -> int:
    if not repSummaries:
        return 0

    for summary in repSummaries:
        if summary["startFrameIndex"] <= frameIndex <= summary["endFrameIndex"]:
            return summary["rep"]

    firstStart = repSummaries[0]["startFrameIndex"]
    if frameIndex < firstStart:
        return 0

    completed = [s["rep"] for s in repSummaries if frameIndex >= s["endFrameIndex"]]
    if completed:
        return max(completed)

    return 0


def _draw_side_panel(
    frameBgr,
    panelX: int,
    panelWidth: int,
    panelTitle: str,
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

    cv2.putText(frameBgr, panelTitle, (left, y), font, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    y += 36
    cv2.line(frameBgr, (left, y), (panelX + panelWidth - 22, y), (210, 210, 210), 1)
    y += 34

    totalReps = len(repSummaries)
    repLabel = f"Rep {max(0, currentRep)} / {totalReps}"
    cv2.putText(frameBgr, repLabel, (left, y), font, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
    y += 34

    if pauseSummary is None:
        cv2.putText(frameBgr, "Tracking...", (left, y), font, 0.62, (70, 70, 70), 2, cv2.LINE_AA)
        y += 32
        for line in _wrap_text("Feedback appears after each rep.", font, 0.55, 1, maxTextWidth):
            cv2.putText(frameBgr, line, (left, y), font, 0.55, (95, 95, 95), 1, cv2.LINE_AA)
            y += 24
        return

    isGoodRep = bool(pauseSummary.get("isGoodRep"))
    rating = str(pauseSummary.get("rating") or "")
    titleColor = (30, 30, 180)
    if isGoodRep or rating in {"strong", "good"}:
        titleColor = (40, 140, 40)
    elif rating == "needs_work":
        titleColor = (20, 110, 190)

    headline = str(
        pauseSummary.get("headline")
        or pauseSummary.get("status")
        or ("Strong rep" if isGoodRep else "Needs adjustment")
    )
    for line in _wrap_text(headline, font, 0.72, 2, maxTextWidth):
        cv2.putText(frameBgr, line, (left, y), font, 0.72, titleColor, 2, cv2.LINE_AA)
        y += 30
    y += 4

    detailLines = pauseSummary.get("detailLines", [])
    if not isinstance(detailLines, list):
        detailLines = []

    for message in detailLines[:3]:
        for line in _wrap_text(str(message), font, 0.57, 1, maxTextWidth):
            cv2.putText(frameBgr, line, (left, y), font, 0.57, (45, 45, 45), 1, cv2.LINE_AA)
            y += 24

    quality = pauseSummary.get("quality")
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


def _create_video_writer(outputPath: Path, fps: float, width: int, height: int):
    """
    Prefer browser-friendly H.264 on Windows (MSMF), fallback to mp4v.
    """
    # Windows Media Foundation path: tends to produce browser-playable MP4 (H.264).
    writer = cv2.VideoWriter(
        str(outputPath),
        cv2.CAP_MSMF,
        cv2.VideoWriter_fourcc(*"H264"),
        fps,
        (width, height),
    )
    if writer.isOpened():
        return writer

    # Fallback path used before (may not be browser-decodable in all browsers).
    writer = cv2.VideoWriter(
        str(outputPath),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    return writer


def create_exercise_feedback_video(
    videoPath: str,
    poseFrames: List[Any],
    outputPath: str,
    panelTitle: str,
    repSummaries: List[Dict[str, Any]],
    pauseSeconds: float = 4.0,
    minVisibility: float = 0.4,
) -> Optional[str]:
    """
    Generic annotated exercise feedback video.

    repSummaries items should include:
      - rep
      - startFrameIndex
      - endFrameIndex
      - status
      - detailLines
      - isGoodRep
    Optional:
      - quality
    """
    if not poseFrames:
        return None

    metadata = get_video_metadata(videoPath)
    fps = float(metadata.get("fps") or 0.0)
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)

    if fps <= 0 or width <= 0 or height <= 0:
        return None

    repByEndFrame: Dict[int, Dict[str, Any]] = {}
    for summary in repSummaries:
        pauseFrame = summary.get("pauseFrameIndex")
        if not isinstance(pauseFrame, int):
            pauseFrame = summary.get("endFrameIndex")
        if isinstance(pauseFrame, int):
            repByEndFrame[pauseFrame] = summary

    pauseFrames = max(0, int(round(fps * pauseSeconds)))
    panelWidth = max(280, int(round(width * 0.38)))
    outputWidth = width + panelWidth

    outputP = Path(outputPath)
    outputP.parent.mkdir(parents=True, exist_ok=True)

    writer = _create_video_writer(
        outputPath=outputP,
        fps=fps,
        width=outputWidth,
        height=height,
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
                panelTitle=panelTitle,
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
