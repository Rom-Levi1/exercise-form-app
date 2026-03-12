from typing import Any, Dict, List

from backend.feedback.common import generic_rep_breakdown, rep_issue_reps, score_to_rating, title_from_rating


def build_front_shoulder_press_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]
    if analysis.get("status") != "success":
        return {
            "overall": {
                "title": "Shoulder press analysis failed",
                "rating": "poor",
                "summary": analysis.get("message")
                or "The front-view shoulder press video could not be analyzed.",
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

    bend_count = rep_summary.get("bottomBendIssueCount") or 0
    lockout_count = rep_summary.get("lockoutIssueCount") or 0
    reach_count = rep_summary.get("topReachIssueCount") or 0
    symmetry_count = rep_summary.get("symmetryIssueCount") or 0

    highlights: List[Dict[str, Any]] = []
    if bend_count:
        highlights.append(
            {
                "title": "Bottom position is too shallow",
                "severity": "medium",
                "details": f"You did not lower far enough before pressing in {bend_count} of {rep_count} rep(s).",
                "cue": "Start each rep from a deeper loaded position before driving up.",
                "reps": rep_issue_reps(rep_feedback, ("bottom_bend_shallow",)),
            }
        )
    if lockout_count:
        highlights.append(
            {
                "title": "Finish the press fully overhead",
                "severity": "medium",
                "details": f"Top lockout looked incomplete in {lockout_count} of {rep_count} rep(s).",
                "cue": "Reach tall at the top and finish each rep fully before lowering.",
                "reps": rep_issue_reps(rep_feedback, ("lockout_incomplete",)),
            }
        )
    if reach_count:
        highlights.append(
            {
                "title": "Top reach stays too low",
                "severity": "medium",
                "details": f"You did not reach a full overhead height in {reach_count} of {rep_count} rep(s).",
                "cue": "Press higher at the top while keeping the ribcage controlled.",
                "reps": rep_issue_reps(rep_feedback, ("top_reach_low",)),
            }
        )
    if symmetry_count:
        highlights.append(
            {
                "title": "Arms finish unevenly overhead",
                "severity": "medium",
                "details": f"Top position symmetry broke down in {symmetry_count} of {rep_count} rep(s).",
                "cue": "Drive both hands up evenly and finish at the same height.",
                "reps": rep_issue_reps(rep_feedback, ("top_symmetry_high_imbalance", "top_symmetry_mild_imbalance")),
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Front-view shoulder press looks solid",
                "severity": None,
                "details": f"Depth, overhead finish, and left-right balance looked solid across {rep_count} rep(s).",
                "cue": "Keep the same pressing rhythm and balanced finish overhead.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _summary(rep_count, bend_count, lockout_count, reach_count, symmetry_count),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(rep_count: int, bend_count: int, lockout_count: int, reach_count: int, symmetry_count: int) -> str:
    if not any((bend_count, lockout_count, reach_count, symmetry_count)):
        return f"{rep_count} shoulder press rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if bend_count:
        parts.append(f"bottom position in {bend_count} rep(s)")
    if lockout_count:
        parts.append(f"lockout in {lockout_count} rep(s)")
    if reach_count:
        parts.append(f"top height in {reach_count} rep(s)")
    if symmetry_count:
        parts.append(f"overhead symmetry in {symmetry_count} rep(s)")
    return f"{rep_count} shoulder press rep(s) were analyzed. The main areas to clean up were " + ", ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "bottom_bend_shallow" in issues:
            item["label"] = "Shallow start"
            item["details"] = "This rep did not start from a deep enough loaded position."
        elif "lockout_incomplete" in issues:
            item["label"] = "Incomplete lockout"
            item["details"] = "The rep was not fully finished overhead."
        elif "top_reach_low" in issues:
            item["label"] = "Low top reach"
            item["details"] = "You did not press high enough at the top."
        elif "top_symmetry_high_imbalance" in issues or "top_symmetry_mild_imbalance" in issues:
            item["label"] = "Uneven overhead finish"
            item["details"] = "One side finished higher or sooner than the other."
    return breakdown
