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

    height_reps = rep_issue_reps(rep_feedback, ("height_incomplete",))
    asymmetry_reps = rep_issue_reps(rep_feedback, ("pull_asymmetry",))

    highlights: List[Dict[str, Any]] = []
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
                "title": "Pulling pattern is uneven",
                "severity": "medium",
                "details": f"Left and right sides did not move evenly in {len(asymmetry_reps)} rep(s).",
                "cue": "Keep both sides pulling with the same timing and range near the top.",
                "reps": asymmetry_reps,
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Pull-up rhythm looks solid",
                "severity": None,
                "details": f"Top position and left-right balance looked solid across {rep_count} rep(s).",
                "cue": "Keep the same pull height and even rhythm from side to side.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _summary(rep_count, len(height_reps), len(asymmetry_reps)),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(rep_count: int, height_count: int, asymmetry_count: int) -> str:
    if not any((height_count, asymmetry_count)):
        return f"{rep_count} pull-up rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if height_count:
        parts.append(f"top position in {height_count} rep(s)")
    if asymmetry_count:
        parts.append(f"left-right balance in {asymmetry_count} rep(s)")
    return f"{rep_count} pull-up rep(s) were analyzed. The main areas to clean up were " + " and ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "height_incomplete" in issues:
            item["label"] = "Low top position"
            item["details"] = "This rep did not reach a high enough top position."
        elif "pull_asymmetry" in issues:
            item["label"] = "Uneven pull"
            item["details"] = "One side moved differently from the other near the top."
    return breakdown
