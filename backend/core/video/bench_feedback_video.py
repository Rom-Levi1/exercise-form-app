from typing import Any, Dict, List, Optional

from backend.core.video.exercise_feedback_video import (
    build_rep_summaries_from_analysis_result,
    build_rep_summaries_from_text_feedback,
    create_exercise_feedback_video,
)


def _issue_code_to_message(issueCode: str) -> str:
    mapping = {
        "rom_incomplete": "Use a fuller range of motion.",
        "bar_path_drift": "Keep the bar path more consistent.",
        "wrist_elbow_stacking": "Keep wrists stacked over elbows.",
        "elbow_tuck_off": "Keep elbow tuck in a better range.",
        "press_asymmetry": "Press more evenly with both arms.",
        "bar_off_center": "Keep the bar path centered.",
    }
    return mapping.get(issueCode, issueCode.replace("_", " "))


def create_bench_feedback_video(
    videoPath: str,
    poseFrames: List[Any],
    analysisResult: Dict[str, Any],
    outputPath: str,
    panelTitle: str,
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
            positiveDetailLines=["Range of motion and rep control looked good."],
            repKeyCandidates=("repIndex", "rep"),
        )

    return create_exercise_feedback_video(
        videoPath=videoPath,
        poseFrames=poseFrames,
        outputPath=outputPath,
        panelTitle=panelTitle,
        repSummaries=repSummaries,
        pauseSeconds=pauseSeconds,
        minVisibility=minVisibility,
    )
