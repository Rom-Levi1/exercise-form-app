from typing import Any, Dict, List

from backend.feedback.common import generic_rep_breakdown, rep_issue_reps, score_to_rating, title_from_rating


def build_side_bicep_curl_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]
    if analysis.get("status") != "success":
        return _failed(analysis, warnings)

    rep_count = analysis.get("repCount") or 0
    score = analysis.get("summaryScore")
    rating = score_to_rating(score if isinstance(score, (int, float)) else None)
    rep_feedback = analysis.get("repFeedback") or []

    rom_reps = rep_issue_reps(rep_feedback, ("rom_incomplete",))
    drift_reps = rep_issue_reps(rep_feedback, ("elbow_drift",))

    highlights: List[Dict[str, Any]] = []
    if rom_reps:
        highlights.append(
            {
                "title": "Curl range is incomplete",
                "severity": "high" if len(rom_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"You missed a full curl range in {len(rom_reps)} of {rep_count} rep(s).",
                "cue": "Open fully at the bottom and finish the curl higher at the top.",
                "reps": rom_reps,
            }
        )
    if drift_reps:
        highlights.append(
            {
                "title": "Elbow drifts away from the torso",
                "severity": "medium",
                "details": f"Elbow position moved too much in {len(drift_reps)} rep(s).",
                "cue": "Pin the elbow closer to your side and let the forearm do the work.",
                "reps": drift_reps,
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Side-view curls look controlled",
                "severity": None,
                "details": f"Range of motion and elbow control looked solid across {rep_count} rep(s).",
                "cue": "Keep the same tempo and elbow position on the next set.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _summary(rep_count, len(rom_reps), len(drift_reps)),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(rep_count: int, rom_count: int, drift_count: int) -> str:
    if not any((rom_count, drift_count)):
        return f"{rep_count} curl rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if rom_count:
        parts.append(f"range in {rom_count} rep(s)")
    if drift_count:
        parts.append(f"elbow position in {drift_count} rep(s)")
    return f"{rep_count} curl rep(s) were analyzed. The main areas to clean up were " + " and ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "rom_incomplete" in issues:
            item["label"] = "Shortened curl"
            item["details"] = "This rep did not use a full curling range."
        elif "elbow_drift" in issues:
            item["label"] = "Elbow drift"
            item["details"] = "Your elbow moved away from its starting position."
    return breakdown


def _failed(analysis: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    return {
        "overall": {
            "title": "Bicep curl analysis failed",
            "rating": "poor",
            "summary": analysis.get("message") or "The side-view bicep curl video could not be analyzed.",
        },
        "highlights": [],
        "repBreakdown": [],
        "warnings": warnings,
    }
