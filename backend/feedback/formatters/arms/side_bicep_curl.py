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

    bottom_reps = rep_issue_reps(rep_feedback, ("bottom_position_incomplete",))
    top_reps = rep_issue_reps(rep_feedback, ("top_position_incomplete",))
    rom_reps = rep_issue_reps(rep_feedback, ("rom_incomplete",))
    drift_reps = rep_issue_reps(rep_feedback, ("elbow_drift",))
    upper_arm_reps = rep_issue_reps(rep_feedback, ("upper_arm_instability",))

    highlights: List[Dict[str, Any]] = []
    if bottom_reps:
        highlights.append(
            {
                "title": "Bottom position is too short",
                "severity": "medium",
                "details": f"You did not lower fully before curling in {len(bottom_reps)} of {rep_count} rep(s).",
                "cue": "Open the elbow more at the bottom before starting the next curl.",
                "reps": bottom_reps,
            }
        )
    if top_reps:
        highlights.append(
            {
                "title": "Top squeeze is too low",
                "severity": "medium",
                "details": f"You did not finish the curl high enough in {len(top_reps)} of {rep_count} rep(s).",
                "cue": "Bring the weight higher at the top without letting the elbow drift forward.",
                "reps": top_reps,
            }
        )
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
    if upper_arm_reps:
        highlights.append(
            {
                "title": "Upper arm moves too much",
                "severity": "medium",
                "details": f"The upper arm shifted or rotated too much in {len(upper_arm_reps)} rep(s).",
                "cue": "Keep the upper arm more fixed and let the forearm create the curl.",
                "reps": upper_arm_reps,
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
            "summary": _summary(
                rep_count,
                len(bottom_reps),
                len(top_reps),
                len(rom_reps),
                len(drift_reps),
                len(upper_arm_reps),
            ),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(
    rep_count: int,
    bottom_count: int,
    top_count: int,
    rom_count: int,
    drift_count: int,
    upper_arm_count: int,
) -> str:
    if not any((bottom_count, top_count, rom_count, drift_count, upper_arm_count)):
        return f"{rep_count} curl rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if bottom_count:
        parts.append(f"bottom position in {bottom_count} rep(s)")
    if top_count:
        parts.append(f"top position in {top_count} rep(s)")
    if rom_count:
        parts.append(f"range in {rom_count} rep(s)")
    if drift_count:
        parts.append(f"elbow position in {drift_count} rep(s)")
    if upper_arm_count:
        parts.append(f"upper-arm control in {upper_arm_count} rep(s)")
    return f"{rep_count} curl rep(s) were analyzed. The main areas to clean up were " + ", ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    issue_details_map = {
        "bottom_position_incomplete": (
            "Short bottom",
            "You did not lower enough before starting the curl.",
        ),
        "top_position_incomplete": (
            "Low top",
            "You did not finish the curl high enough.",
        ),
        "rom_incomplete": (
            "Shortened curl",
            "This rep did not use a full curling range.",
        ),
        "elbow_drift": (
            "Elbow drift",
            "Your elbow moved away from its starting position.",
        ),
        "upper_arm_instability": (
            "Upper arm unstable",
            "The upper arm shifted too much during the curl.",
        ),
    }

    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if not isinstance(issues, list) or not issues:
            continue

        rep_labels: List[str] = []
        rep_details: List[str] = []
        for issue in issues:
            if not isinstance(issue, str):
                continue
            mapped = issue_details_map.get(issue)
            if mapped is None:
                continue
            label, detail = mapped
            rep_labels.append(label)
            rep_details.append(detail)

        if rep_labels:
            item["label"] = " | ".join(rep_labels)
            item["details"] = " ".join(rep_details)
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
