from typing import Any, Dict, List

from backend.feedback.common import generic_rep_breakdown, rep_issue_reps, score_to_rating, title_from_rating


def build_front_squat_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]
    if analysis.get("status") != "success":
        return {
            "overall": {
                "title": "Front squat analysis failed",
                "rating": "poor",
                "summary": analysis.get("message")
                or "The front-view squat video could not be analyzed.",
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
    stance_issue_count = rep_summary.get("stanceIssueCount") or 0
    symmetry_issue_count = rep_summary.get("symmetryIssueCount") or 0

    highlights: List[Dict[str, Any]] = []

    narrow_reps = rep_issue_reps(rep_feedback, ("stance_too_narrow",))
    wide_reps = rep_issue_reps(rep_feedback, ("stance_too_wide",))
    imbalance_reps = rep_issue_reps(
        rep_feedback,
        ("symmetry_high_imbalance", "symmetry_mild_imbalance"),
    )

    if narrow_reps:
        highlights.append(
            {
                "title": "Stance looks too narrow",
                "severity": "medium",
                "details": f"Your stance looked too narrow in {len(narrow_reps)} of {rep_count} rep(s).",
                "cue": "Set your feet a bit wider before the descent and keep that width consistent.",
                "reps": narrow_reps,
            }
        )
    if wide_reps:
        highlights.append(
            {
                "title": "Stance looks very wide",
                "severity": "low",
                "details": f"Your stance looked too wide in {len(wide_reps)} of {rep_count} rep(s).",
                "cue": "Bring your feet slightly closer together and keep the setup balanced.",
                "reps": wide_reps,
            }
        )

    if imbalance_reps:
        highlights.append(
            {
                "title": "Left and right sides are uneven",
                "severity": "medium" if symmetry_issue_count >= max(1, rep_count / 2) else "low",
                "details": f"Left-right balance drifted in {len(imbalance_reps)} of {rep_count} rep(s).",
                "cue": "Keep your chest centered and distribute pressure more evenly through both feet.",
                "reps": imbalance_reps,
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Front-view squat set looks solid",
                "severity": None,
                "details": f"Stance width and left-right balance looked stable across {rep_count} rep(s).",
                "cue": "Keep using the same setup and centered balance.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _build_summary(rep_count, stance_issue_count, symmetry_issue_count),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _build_summary(rep_count: int, stance_issue_count: int, symmetry_issue_count: int) -> str:
    if stance_issue_count == 0 and symmetry_issue_count == 0:
        return f"{rep_count} front-view squat rep(s) were analyzed and no major form issues were flagged."

    parts: List[str] = []
    if stance_issue_count:
        parts.append(f"stance width in {stance_issue_count} rep(s)")
    if symmetry_issue_count:
        parts.append(f"left-right balance in {symmetry_issue_count} rep(s)")

    return (
        f"{rep_count} front-view squat rep(s) were analyzed. "
        f"The main areas to clean up were {', '.join(parts)}."
    )


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "stance_too_narrow" in issues:
            item["label"] = "Narrow stance"
            item["details"] = "This rep started and stayed narrower than the target stance."
        elif "stance_too_wide" in issues:
            item["label"] = "Wide stance"
            item["details"] = "This rep used a wider stance than the target range."
        elif "symmetry_high_imbalance" in issues or "symmetry_mild_imbalance" in issues:
            item["label"] = "Left-right imbalance"
            item["details"] = "Weight or body position shifted unevenly during this rep."
    return breakdown
