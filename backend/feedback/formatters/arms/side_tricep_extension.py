from typing import Any, Dict, List

from backend.feedback.common import generic_rep_breakdown, rep_issue_reps, score_to_rating, title_from_rating


def build_side_tricep_extension_feedback(analysis: Dict[str, Any]) -> Dict[str, Any]:
    warnings = [warning for warning in (analysis.get("warnings") or []) if isinstance(warning, str)]
    if analysis.get("status") != "success":
        return {
            "overall": {
                "title": "Tricep extension analysis failed",
                "rating": "poor",
                "summary": analysis.get("message")
                or "The side-view tricep extension video could not be analyzed.",
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
    elbow_reps = rep_issue_reps(rep_feedback, ("elbow_drift",))
    upper_arm_reps = rep_issue_reps(rep_feedback, ("upper_arm_instability",))

    highlights: List[Dict[str, Any]] = []
    if rom_reps:
        highlights.append(
            {
                "title": "Range of motion is incomplete",
                "severity": "high" if len(rom_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"You missed a full extension range in {len(rom_reps)} of {rep_count} rep(s).",
                "cue": "Bend more at the bottom and finish the extension fully at the top.",
                "reps": rom_reps,
            }
        )
    if elbow_reps:
        highlights.append(
            {
                "title": "Elbow drifts too much",
                "severity": "medium",
                "details": f"The elbow moved more than expected in {len(elbow_reps)} rep(s).",
                "cue": "Keep the elbow more fixed so the forearm drives the movement.",
                "reps": elbow_reps,
            }
        )
    if upper_arm_reps:
        highlights.append(
            {
                "title": "Upper arm loses stability",
                "severity": "medium",
                "details": f"The upper arm rotated or shifted too much in {len(upper_arm_reps)} rep(s).",
                "cue": "Keep the upper arm quieter and let the elbow extension create the motion.",
                "reps": upper_arm_reps,
            }
        )

    if not highlights:
        highlights.append(
            {
                "title": "Side-view tricep extensions look controlled",
                "severity": None,
                "details": f"Range of motion, elbow position, and upper-arm stability looked solid across {rep_count} rep(s).",
                "cue": "Keep the same elbow position and extension tempo.",
                "reps": [],
            }
        )

    return {
        "overall": {
            "title": title_from_rating(rating),
            "rating": rating,
            "summary": _summary(rep_count, len(rom_reps), len(elbow_reps), len(upper_arm_reps)),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(rep_count: int, rom_count: int, elbow_count: int, upper_arm_count: int) -> str:
    if not any((rom_count, elbow_count, upper_arm_count)):
        return f"{rep_count} tricep extension rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if rom_count:
        parts.append(f"range in {rom_count} rep(s)")
    if elbow_count:
        parts.append(f"elbow position in {elbow_count} rep(s)")
    if upper_arm_count:
        parts.append(f"upper-arm stability in {upper_arm_count} rep(s)")
    return f"{rep_count} tricep extension rep(s) were analyzed. The main areas to clean up were " + ", ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    for item, rep in zip(breakdown, rep_feedback):
        issues = rep.get("issues") or []
        if "rom_incomplete" in issues:
            item["label"] = "Shortened extension"
            item["details"] = "This rep did not use a full extension range."
        elif "elbow_drift" in issues:
            item["label"] = "Elbow drift"
            item["details"] = "Your elbow moved away from a stable path."
        elif "upper_arm_instability" in issues:
            item["label"] = "Upper arm unstable"
            item["details"] = "The upper arm shifted too much during the rep."
    return breakdown
