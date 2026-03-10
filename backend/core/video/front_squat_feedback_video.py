from typing import Any, Dict, List, Optional

from backend.core.video.exercise_feedback_video import create_exercise_feedback_video


def create_front_squat_feedback_video(
    videoPath: str,
    poseFrames: List[Any],
    analysisResult: Dict[str, Any],
    outputPath: str,
    pauseSeconds: float = 4.0,
    minVisibility: float = 0.4,
) -> Optional[str]:
    repFeedback = analysisResult.get("repFeedback", [])
    clipSummary = repFeedback[0] if isinstance(repFeedback, list) and repFeedback else {}
    issues = clipSummary.get("issues", []) if isinstance(clipSummary, dict) else []
    checks = clipSummary.get("checks", {}) if isinstance(clipSummary, dict) else {}
    stanceWidth = checks.get("stanceWidth", {}) if isinstance(checks, dict) else {}
    symmetry = checks.get("symmetry", {}) if isinstance(checks, dict) else {}

    detailLines: List[str] = []
    if not isinstance(issues, list) or len(issues) == 0:
        detailLines.append("Stance width and symmetry looked good.")
    else:
        if "stance_width_issue" in issues:
            classification = stanceWidth.get("classification")
            if classification == "too_narrow":
                detailLines.append("Widen your stance slightly.")
            elif classification == "too_wide":
                detailLines.append("Bring your stance in slightly.")
            else:
                detailLines.append("Adjust your stance width.")

        if "asymmetry_issue" in issues:
            detailLines.append("Keep your left and right sides more balanced.")

    if not detailLines:
        detailLines.append("Front-view squat summary ready.")

    lastFrameIndex = max(0, len(poseFrames) - 1)
    repSummaries = [
        {
            "rep": 1,
            "startFrameIndex": 0,
            "endFrameIndex": lastFrameIndex,
            "quality": clipSummary.get("quality"),
            "issues": issues if isinstance(issues, list) else [],
            "status": "Good clip." if len(issues or []) == 0 else "Needs work:",
            "detailLines": detailLines[:3],
            "isGoodRep": len(issues or []) == 0,
        }
    ]

    return create_exercise_feedback_video(
        videoPath=videoPath,
        poseFrames=poseFrames,
        outputPath=outputPath,
        panelTitle="Squats",
        repSummaries=repSummaries,
        pauseSeconds=pauseSeconds,
        minVisibility=minVisibility,
    )
