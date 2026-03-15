from typing import Any, Dict, List, Optional

from backend.core.video.exercise_feedback_video import (
    build_rep_summaries_from_analysis_result,
    build_rep_summaries_from_text_feedback,
    create_exercise_feedback_video,
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


def create_squat_feedback_video(
    videoPath: str,
    poseFrames: List[Any],
    analysisResult: Dict[str, Any],
    outputPath: str,
    textFeedback: Optional[Dict[str, Any]] = None,
    pauseSeconds: float = 4.0,
    minVisibility: float = 0.4,
) -> Optional[str]:
    repSummaries = build_rep_summaries_from_text_feedback(analysisResult, textFeedback)
    if not repSummaries:
        repSummaries = build_rep_summaries_from_analysis_result(
            analysisResult,
            issueMessageResolver=_issue_code_to_message,
            positiveStatus="Good rep.",
            positiveDetailLines=["Depth, torso angle, and lockout looked good."],
        )

    return create_exercise_feedback_video(
        videoPath=videoPath,
        poseFrames=poseFrames,
        outputPath=outputPath,
        panelTitle="Squats",
        repSummaries=repSummaries,
        pauseSeconds=pauseSeconds,
        minVisibility=minVisibility,
    )
