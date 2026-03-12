from typing import Any, Dict, List

from backend.feedback.common import (
    generic_rep_breakdown,
    rep_issue_reps,
    score_to_rating,
    title_from_rating,
)


def build_side_squat_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    status = analysis.get("status")
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]

    if status != "success":
        return {
            "overall": {
                "title": "Side squat analysis failed",
                "rating": "poor",
                "summary": analysis.get("message") or "The side-view squat video could not be analyzed.",
            },
            "highlights": [],
            "repBreakdown": [],
            "warnings": warnings,
        }

    rep_count = analysis.get("repCount") or 0
    score = analysis.get("summaryScore")
    rating = score_to_rating(score if isinstance(score, (int, float)) else None)
    rep_feedback = analysis.get("repFeedback") or []
    rep_summary = (analysis.get("metrics") or {}).get("repCheckSummary") or {}

    depth_count = rep_summary.get("depthIssueCount") or 0
    torso_count = rep_summary.get("torsoLeanIssueCount") or 0
    lockout_count = rep_summary.get("lockoutIssueCount") or 0

    if rep_count == 0:
        return {
            "overall": {
                "title": "No clear squat reps detected",
                "rating": "poor",
                "summary": analysis.get("message")
                or "Try a cleaner side view with the full body visible from start to finish.",
            },
            "highlights": [],
            "repBreakdown": [],
            "warnings": warnings,
        }

    highlights: List[Dict[str, Any]] = []

    if depth_count:
        highlights.append(
            {
                "title": "Depth is inconsistent",
                "severity": "high" if depth_count >= rep_count / 2 else "medium",
                "details": f"Depth looked shallow in {depth_count} of {rep_count} rep(s).",
                "cue": "Sit lower under control and reach a more consistent bottom position.",
                "reps": rep_issue_reps(rep_feedback, ("depth_high", "depth_moderate")),
            }
        )

    if torso_count:
        highlights.append(
            {
                "title": "Torso leans forward too much",
                "severity": "high" if torso_count >= rep_count / 2 else "medium",
                "details": f"Forward torso lean increased in {torso_count} of {rep_count} rep(s).",
                "cue": "Brace harder before descending and keep the chest taller through the bottom.",
                "reps": rep_issue_reps(
                    rep_feedback,
                    ("torso_lean_excessive", "torso_lean_moderate"),
                ),
            }
        )

    if lockout_count:
        highlights.append(
            {
                "title": "Finish the rep more fully",
                "severity": "medium",
                "details": f"Top lockout looked incomplete in {lockout_count} of {rep_count} rep(s).",
                "cue": "Stand all the way tall and finish each ascent before starting the next rep.",
                "reps": rep_issue_reps(rep_feedback, ("lockout_incomplete",)),
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Strong side-view squat set",
                "severity": None,
                "details": f"Depth, torso position, and lockout all looked solid across {rep_count} rep(s).",
                "cue": "Keep the same balance, bracing, and control on the next set.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _build_summary(rep_count, depth_count, torso_count, lockout_count),
        },
        "highlights": highlights,
        "repBreakdown": _build_rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _build_summary(
    rep_count: int,
    depth_count: int,
    torso_count: int,
    lockout_count: int,
) -> str:
    if depth_count == 0 and torso_count == 0 and lockout_count == 0:
        return f"{rep_count} squat rep(s) were analyzed and no major form issues were flagged."

    issue_parts: List[str] = []
    if depth_count:
        issue_parts.append(f"depth in {depth_count} rep(s)")
    if torso_count:
        issue_parts.append(f"torso position in {torso_count} rep(s)")
    if lockout_count:
        issue_parts.append(f"lockout in {lockout_count} rep(s)")

    return (
        f"{rep_count} squat rep(s) were analyzed. "
        f"The main areas to clean up were {', '.join(issue_parts)}."
    )


def _build_rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)

    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "depth_high" in issues or "depth_moderate" in issues:
            item["label"] = "Shallow depth"
            item["details"] = "This rep did not reach the target squat depth."
        elif "torso_lean_excessive" in issues or "torso_lean_moderate" in issues:
            item["label"] = "Forward torso lean"
            item["details"] = "Your torso tipped forward too much during the rep."
        elif "lockout_incomplete" in issues:
            item["label"] = "Incomplete lockout"
            item["details"] = "The rep was not fully finished at the top."

    return breakdown
