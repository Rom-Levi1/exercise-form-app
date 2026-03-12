from typing import Any, Dict, List

from backend.feedback.common import generic_rep_breakdown, rep_issue_reps, score_to_rating, title_from_rating


def build_side_bench_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]
    if analysis.get("status") != "success":
        return {
            "overall": {
                "title": "Side bench analysis failed",
                "rating": "poor",
                "summary": analysis.get("message") or "The side-view bench video could not be analyzed.",
            },
            "highlights": [],
            "repBreakdown": [],
            "warnings": warnings,
        }

    rep_count = analysis.get("repCount") or 0
    score = analysis.get("summaryScore")
    rating = score_to_rating(score if isinstance(score, (int, float)) else None)
    rep_feedback = analysis.get("repFeedback") or []

    rom_reps = rep_issue_reps(rep_feedback, ("rom_incomplete",))
    path_reps = rep_issue_reps(rep_feedback, ("bar_path_drift",))
    stacking_reps = rep_issue_reps(rep_feedback, ("wrist_elbow_stacking",))
    tuck_reps = rep_issue_reps(rep_feedback, ("elbow_tuck_off",))

    highlights: List[Dict[str, Any]] = []
    if rom_reps:
        highlights.append(
            {
                "title": "Use a fuller range",
                "severity": "high" if len(rom_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"Range of motion looked incomplete in {len(rom_reps)} of {rep_count} rep(s).",
                "cue": "Lower under control and finish each rep fully before the next one.",
                "reps": rom_reps,
            }
        )
    if path_reps:
        highlights.append(
            {
                "title": "Hand path drifts too much",
                "severity": "medium",
                "details": f"Your hand path wandered more than expected in {len(path_reps)} rep(s).",
                "cue": "Keep the pressing path straighter from the bottom to the top.",
                "reps": path_reps,
            }
        )
    if stacking_reps:
        highlights.append(
            {
                "title": "Forearm stacking breaks down",
                "severity": "medium",
                "details": f"Your wrist moved out of line with the elbow in {len(stacking_reps)} rep(s).",
                "cue": "Keep the wrist stacked over the elbow as you press.",
                "reps": stacking_reps,
            }
        )
    if tuck_reps:
        highlights.append(
            {
                "title": "Elbow path is inconsistent",
                "severity": "low",
                "details": f"Elbow position moved outside the target path in {len(tuck_reps)} rep(s).",
                "cue": "Keep the elbow path more repeatable from rep to rep.",
                "reps": tuck_reps,
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Side-view bench set looks solid",
                "severity": None,
                "details": f"Range, hand path, and arm positioning looked solid across {rep_count} rep(s).",
                "cue": "Keep the same touch point and steady press path.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _summary(rep_count, len(rom_reps), len(path_reps), len(stacking_reps), len(tuck_reps)),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(rep_count: int, rom_count: int, path_count: int, stacking_count: int, tuck_count: int) -> str:
    if not any((rom_count, path_count, stacking_count, tuck_count)):
        return f"{rep_count} side-view bench rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if rom_count:
        parts.append(f"range in {rom_count} rep(s)")
    if path_count:
        parts.append(f"hand path in {path_count} rep(s)")
    if stacking_count:
        parts.append(f"wrist-to-elbow stacking in {stacking_count} rep(s)")
    if tuck_count:
        parts.append(f"elbow path in {tuck_count} rep(s)")
    return f"{rep_count} side-view bench rep(s) were analyzed. The main areas to clean up were " + ", ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "rom_incomplete" in issues:
            item["label"] = "Incomplete range"
            item["details"] = "This rep did not use a full pressing range."
        elif "bar_path_drift" in issues:
            item["label"] = "Drifting path"
            item["details"] = "Your hand path drifted horizontally during the rep."
        elif "wrist_elbow_stacking" in issues:
            item["label"] = "Stacking breakdown"
            item["details"] = "The wrist moved out of line with the elbow."
        elif "elbow_tuck_off" in issues:
            item["label"] = "Elbow path off"
            item["details"] = "The elbow moved outside the intended pressing path."
    return breakdown
