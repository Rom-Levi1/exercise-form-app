from typing import Any, Dict, List

from backend.feedback.common import generic_rep_breakdown, rep_issue_reps, score_to_rating, title_from_rating


def build_pullup_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]
    if analysis.get("status") != "success":
        return {
            "overall": {
                "title": "Pull-up analysis failed",
                "rating": "poor",
                "summary": analysis.get("message") or "The pull-up video could not be analyzed.",
            },
            "highlights": [],
            "repBreakdown": [],
            "warnings": warnings,
        }

    rep_count = analysis.get("repCount") or 0
    score = analysis.get("summaryScore")
    rating = score_to_rating(score if isinstance(score, (int, float)) else None)
    rep_feedback = analysis.get("repFeedback") or []

    bottom_reps = rep_issue_reps(rep_feedback, ("bottom_incomplete",))
    height_reps = rep_issue_reps(rep_feedback, ("height_incomplete",))
    asymmetry_reps = rep_issue_reps(rep_feedback, ("pull_asymmetry",))
    highlights: List[Dict[str, Any]] = []
    if bottom_reps:
        highlights.append(
            {
                "title": "Bottom hang is not deep enough",
                "severity": "high" if len(bottom_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"You did not return to a deep enough bottom position in {len(bottom_reps)} of {rep_count} rep(s).",
                "cue": "Lower all the way into a near-straight-arm hang before starting the next rep.",
                "reps": bottom_reps,
            }
        )
    if height_reps:
        highlights.append(
            {
                "title": "Top position is not high enough",
                "severity": "high" if len(height_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"You did not reach the target top position in {len(height_reps)} of {rep_count} rep(s).",
                "cue": "Pull a little higher before ending the rep.",
                "reps": height_reps,
            }
        )
    if asymmetry_reps:
        highlights.append(
            {
                "title": "Pull looks slightly uneven",
                "severity": "low",
                "details": f"Left and right elbows reached the top a bit unevenly in {len(asymmetry_reps)} of {rep_count} rep(s).",
                "cue": "Try to keep both arms finishing the pull at the same time near the top.",
                "reps": asymmetry_reps,
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Pull-up range looks solid",
                "severity": None,
                "details": f"Bottom hang, top position, and top balance looked solid across {rep_count} rep(s).",
                "cue": "Keep using the same full bottom-to-top range each rep.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _summary(rep_count, len(bottom_reps), len(height_reps), len(asymmetry_reps)),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(rep_count: int, bottom_count: int, height_count: int, asymmetry_count: int) -> str:
    if not any((bottom_count, height_count, asymmetry_count)):
        return f"{rep_count} pull-up rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if bottom_count:
        parts.append(f"bottom position in {bottom_count} rep(s)")
    if height_count:
        parts.append(f"top position in {height_count} rep(s)")
    if asymmetry_count:
        parts.append(f"top balance in {asymmetry_count} rep(s)")
    return (
        f"{rep_count} pull-up rep(s) were analyzed. "
        f"The main areas to clean up were " + ", ".join(parts) + "."
    )


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "bottom_incomplete" in issues and "height_incomplete" in issues:
            item["label"] = "Short ROM"
            item["details"] = "This rep did not reach a deep enough bottom or a high enough top."
        elif "bottom_incomplete" in issues:
            item["label"] = "Shallow bottom hang"
            item["details"] = "This rep did not return to a deep enough bottom position."
        elif "pull_asymmetry" in issues:
            item["label"] = "Uneven top finish"
            item["details"] = "Your left and right elbows did not finish the pull evenly at the top."
        elif "height_incomplete" in issues:
            item["label"] = "Low top position"
            item["details"] = "This rep did not reach a high enough top position."
    return breakdown
