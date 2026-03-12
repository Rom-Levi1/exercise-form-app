from typing import Any, Callable, Dict

from backend.feedback.common import generic_feedback
from backend.feedback.formatters.arms.side_bicep_curl import build_side_bicep_curl_feedback
from backend.feedback.formatters.arms.side_tricep_extension import (
    build_side_tricep_extension_feedback,
)
from backend.feedback.formatters.back.pullup import build_pullup_feedback
from backend.feedback.formatters.chest.front_bench_press import build_front_bench_feedback
from backend.feedback.formatters.chest.side_bench_press import build_side_bench_feedback
from backend.feedback.formatters.legs.front_squat import build_front_squat_feedback
from backend.feedback.formatters.legs.side_squat import build_side_squat_feedback
from backend.feedback.formatters.shoulders.front_shoulder_press import (
    build_front_shoulder_press_feedback,
)


FeedbackBuilder = Callable[[Dict[str, Any]], Dict[str, Any]]


FORMATTERS: Dict[str, FeedbackBuilder] = {
    "bench_front": build_front_bench_feedback,
    "bench_side": build_side_bench_feedback,
    "bicep_curl_side": build_side_bicep_curl_feedback,
    "pullup_back": build_pullup_feedback,
    "shoulder_press_front": build_front_shoulder_press_feedback,
    "squat_front": build_front_squat_feedback,
    "squat_side": build_side_squat_feedback,
    "tricep_extension_side": build_side_tricep_extension_feedback,
}


def build_text_feedback(exercise: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    formatter = FORMATTERS.get(exercise)
    if formatter is None:
        return generic_feedback(exercise, analysis)
    return formatter(analysis)
