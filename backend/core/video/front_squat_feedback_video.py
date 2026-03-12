from typing import Any, Dict, List, Optional

from backend.core.video.exercise_feedback_video import (
    build_rep_summaries_from_analysis_result,
    create_exercise_feedback_video,
)


FRONT_SQUAT_ISSUE_MESSAGES = {
    "stance_too_narrow": "Widen your stance slightly.",
    "stance_too_wide": "Bring your stance in slightly.",
    "symmetry_high_imbalance": "Keep your left and right sides more balanced.",
    "symmetry_mild_imbalance": "Stay centered and avoid shifting to one side.",
}


def create_front_squat_feedback_video(
    videoPath: str,
    poseFrames: List[Any],
    analysisResult: Dict[str, Any],
    outputPath: str,
    pauseSeconds: float = 4.0,
    minVisibility: float = 0.4,
) -> Optional[str]:
    repSummaries = build_rep_summaries_from_analysis_result(
        analysisResult=analysisResult,
        issueMessageResolver=lambda code: FRONT_SQUAT_ISSUE_MESSAGES.get(
            code,
            str(code).replace("_", " "),
        ),
        positiveStatus="Good rep.",
        positiveDetailLines=[
            "Stance width and left-right balance looked good.",
        ],
    )

    if not repSummaries:
        return None

    return create_exercise_feedback_video(
        videoPath=videoPath,
        poseFrames=poseFrames,
        outputPath=outputPath,
        panelTitle="Squats",
        repSummaries=repSummaries,
        pauseSeconds=pauseSeconds,
        minVisibility=minVisibility,
    )
