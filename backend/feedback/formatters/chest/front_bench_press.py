from typing import Any, Dict, List

from backend.feedback.common import generic_rep_breakdown, rep_issue_reps, score_to_rating, title_from_rating


def build_front_bench_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]
    if analysis.get("status") != "success":
        return _failed_response(analysis, warnings)

    rep_count = analysis.get("repCount") or 0
    score = analysis.get("summaryScore")
    rating = score_to_rating(score if isinstance(score, (int, float)) else None)
    rep_feedback = analysis.get("repFeedback") or []

    rom_reps = rep_issue_reps(rep_feedback, ("rom_incomplete",))
    symmetry_reps = rep_issue_reps(rep_feedback, ("press_asymmetry",))
    centered_reps = rep_issue_reps(rep_feedback, ("bar_off_center",))
    grip_issue = any(
        isinstance(issue, dict) and issue.get("code") == "grip_width_out_of_range"
        for issue in (analysis.get("issues") or [])
    )

    highlights: List[Dict[str, Any]] = []

    if grip_issue:
        highlights.append(
            {
                "title": "Grip width looks off",
                "severity": "low",
                "details": "Your hand spacing stayed outside the target range for this movement.",
                "cue": "Adjust hand placement so both arms start from a more balanced pressing position.",
                "reps": [],
            }
        )

    if rom_reps:
        highlights.append(
            {
                "title": "Use a fuller press range",
                "severity": "high" if len(rom_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"Range of motion looked incomplete in {len(rom_reps)} of {rep_count} rep(s).",
                "cue": "Lower and finish each rep more completely while keeping control.",
                "reps": rom_reps,
            }
        )

    if symmetry_reps:
        highlights.append(
            {
                "title": "Pressing becomes uneven",
                "severity": "medium",
                "details": f"Left and right arm timing drifted apart in {len(symmetry_reps)} rep(s).",
                "cue": "Drive both hands evenly and keep the elbows moving together.",
                "reps": symmetry_reps,
            }
        )

    if centered_reps:
        highlights.append(
            {
                "title": "Press path drifts off center",
                "severity": "medium",
                "details": f"Your hands drifted sideways more than expected in {len(centered_reps)} rep(s).",
                "cue": "Keep the press path centered over your torso from bottom to top.",
                "reps": centered_reps,
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Front-view bench set looks solid",
                "severity": None,
                "details": f"Hand spacing, range of motion, and left-right control looked solid across {rep_count} rep(s).",
                "cue": "Keep the same hand spacing and balanced press tempo.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _summary(rep_count, grip_issue, len(rom_reps), len(symmetry_reps), len(centered_reps)),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(rep_count: int, grip_issue: bool, rom_count: int, symmetry_count: int, centered_count: int) -> str:
    if not any((grip_issue, rom_count, symmetry_count, centered_count)):
        return f"{rep_count} front-view bench rep(s) were analyzed and no major form issues were flagged."

    parts: List[str] = []
    if grip_issue:
        parts.append("grip width")
    if rom_count:
        parts.append(f"range of motion in {rom_count} rep(s)")
    if symmetry_count:
        parts.append(f"left-right timing in {symmetry_count} rep(s)")
    if centered_count:
        parts.append(f"press path in {centered_count} rep(s)")
    return f"{rep_count} front-view bench rep(s) were analyzed. The main areas to clean up were " + ", ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "rom_incomplete" in issues:
            item["label"] = "Incomplete range"
            item["details"] = "This rep did not reach a full lowering and finishing range."
        elif "press_asymmetry" in issues:
            item["label"] = "Uneven press"
            item["details"] = "Your arms were not moving together evenly through the rep."
        elif "bar_off_center" in issues:
            item["label"] = "Off-center path"
            item["details"] = "Your hand path drifted sideways during the press."
    return breakdown


def _failed_response(analysis: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    return {
        "overall": {
            "title": "Front bench analysis failed",
            "rating": "poor",
            "summary": analysis.get("message") or "The front-view bench video could not be analyzed.",
        },
        "highlights": [],
        "repBreakdown": [],
        "warnings": warnings,
    }
