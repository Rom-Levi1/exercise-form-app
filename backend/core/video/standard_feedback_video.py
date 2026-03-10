from typing import Any, Dict, List, Optional

from backend.core.video.exercise_feedback_video import (
    build_rep_summaries_from_analysis_result,
    create_exercise_feedback_video,
)


def create_standard_feedback_video(
    videoPath: str,
    poseFrames: List[Any],
    analysisResult: Dict[str, Any],
    outputPath: str,
    panelTitle: str,
    issueMessages: Dict[str, str],
    positiveDetailLines: List[str],
    pauseSeconds: float = 4.0,
    minVisibility: float = 0.4,
) -> Optional[str]:
    repSummaries = build_rep_summaries_from_analysis_result(
        analysisResult,
        issueMessageResolver=lambda issueCode: issueMessages.get(
            issueCode,
            issueCode.replace("_", " "),
        ),
        positiveStatus="Good rep.",
        positiveDetailLines=positiveDetailLines,
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
