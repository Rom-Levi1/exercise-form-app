from typing import Any, Dict, Iterable, List, Optional


def normalize_score(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    if 0.0 <= score <= 1.0:
        return score * 100.0
    return score


def score_to_rating(score: Optional[float]) -> str:
    score = normalize_score(score)
    if score is None:
        return "unknown"
    if score >= 90:
        return "strong"
    if score >= 75:
        return "good"
    if score >= 55:
        return "needs_work"
    return "poor"


def title_from_rating(rating: str) -> str:
    return {
        "strong": "Strong form overall",
        "good": "Good form with minor fixes",
        "needs_work": "Decent set with clear fix points",
        "poor": "Form needs attention",
        "unknown": "Feedback unavailable",
    }.get(rating, "Feedback unavailable")


def rep_number(rep: Dict[str, Any]) -> Optional[int]:
    value = rep.get("rep")
    if value is None:
        value = rep.get("repIndex")
    return value if isinstance(value, int) else None


def rep_issue_reps(rep_feedback: Iterable[Dict[str, Any]], prefixes: Iterable[str]) -> List[int]:
    matching_reps: List[int] = []
    prefix_tuple = tuple(prefixes)

    for rep in rep_feedback:
        rep_num = rep_number(rep)
        issues = rep.get("issues") or []
        if rep_num is None or not isinstance(issues, list):
            continue
        if any(isinstance(issue, str) and issue.startswith(prefix_tuple) for issue in issues):
            matching_reps.append(rep_num)

    return matching_reps


def generic_rep_breakdown(rep_feedback: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown: List[Dict[str, Any]] = []

    for rep in rep_feedback:
        rep_num = rep_number(rep)
        if rep_num is None:
            continue

        score = rep.get("quality")
        issues = [issue for issue in (rep.get("issues") or []) if isinstance(issue, str)]
        rating = score_to_rating(score if isinstance(score, (int, float)) else None)

        if not issues:
            label = "Strong rep"
            details = "No major issues were flagged in this rep."
        else:
            rating = "needs_work"
            label = "Needs adjustment"
            details = ", ".join(issue.replace("_", " ") for issue in issues)

        breakdown.append(
            {
                "rep": rep_num,
                "rating": rating,
                "label": label,
                "details": details,
            }
        )

    return breakdown


def generic_feedback(exercise: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    status = analysis.get("status")
    rep_count = analysis.get("repCount") or 0
    score = analysis.get("summaryScore")
    rating = score_to_rating(score if isinstance(score, (int, float)) else None)
    issues = analysis.get("issues") or []
    warnings = analysis.get("warnings") or []

    if status != "success":
        return {
            "overall": {
                "title": "Analysis could not be completed",
                "rating": "poor",
                "summary": analysis.get("message") or "The video could not be analyzed.",
            },
            "highlights": [],
            "repBreakdown": [],
            "warnings": warnings,
        }

    highlights = [
        {
            "title": issue.get("code", "Issue").replace("_", " ").title(),
            "severity": issue.get("severity", "low"),
            "details": issue.get("message", ""),
            "cue": "Review the flagged phase in the feedback video.",
            "reps": [],
        }
        for issue in issues
        if isinstance(issue, dict)
    ]

    if not highlights:
        highlights = [
            {
                "title": "No major issues detected",
                "severity": "low",
                "details": f"{rep_count} rep(s) were analyzed without major top-level issues.",
                "cue": "Keep the same setup and tempo on future sets.",
                "reps": [],
            }
        ]

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": f"{exercise.replace('_', ' ')} analysis completed over {rep_count} rep(s).",
        },
        "highlights": highlights,
        "repBreakdown": generic_rep_breakdown(analysis.get("repFeedback") or []),
        "warnings": [warning for warning in warnings if isinstance(warning, str)],
    }
