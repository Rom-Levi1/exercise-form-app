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

    start_reps = rep_issue_reps(rep_feedback, ("bottom_position_shallow",))
    lockout_reps = rep_issue_reps(rep_feedback, ("lockout_incomplete",))
    rom_reps = rep_issue_reps(rep_feedback, ("rom_incomplete",))
    elbow_reps = rep_issue_reps(rep_feedback, ("elbow_drift",))
    upper_arm_reps = rep_issue_reps(rep_feedback, ("upper_arm_instability",))

    highlights: List[Dict[str, Any]] = []
    if start_reps:
        highlights.append(
            {
                "title": "Start position needs more bend",
                "severity": "medium",
                "details": f"You did not lower into a loaded enough starting position in {len(start_reps)} of {rep_count} rep(s).",
                "cue": "Let the elbow bend more before driving into the extension.",
                "reps": start_reps,
            }
        )
    if lockout_reps:
        highlights.append(
            {
                "title": "Finish the extension more fully",
                "severity": "high" if len(lockout_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"You did not reach full lockout in {len(lockout_reps)} of {rep_count} rep(s).",
                "cue": "Keep extending until the elbow straightens cleanly at the end of the rep.",
                "reps": lockout_reps,
            }
        )
    if rom_reps:
        highlights.append(
            {
                "title": "Range of motion is incomplete",
                "severity": "high" if len(rom_reps) >= max(1, rep_count / 2) else "medium",
                "details": f"The total rep range stayed short in {len(rom_reps)} of {rep_count} rep(s).",
                "cue": "Use a bigger bend-to-lockout range through the whole rep.",
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
            "summary": _summary(
                rep_count,
                len(start_reps),
                len(lockout_reps),
                len(rom_reps),
                len(elbow_reps),
                len(upper_arm_reps),
            ),
        },
        "highlights": highlights,
        "repBreakdown": _rep_breakdown(rep_feedback),
        "warnings": warnings,
    }


def _summary(
    rep_count: int,
    start_count: int,
    lockout_count: int,
    rom_count: int,
    elbow_count: int,
    upper_arm_count: int,
) -> str:
    if not any((start_count, lockout_count, rom_count, elbow_count, upper_arm_count)):
        return f"{rep_count} tricep extension rep(s) were analyzed and no major form issues were flagged."
    parts: List[str] = []
    if start_count:
        parts.append(f"start position in {start_count} rep(s)")
    if lockout_count:
        parts.append(f"lockout in {lockout_count} rep(s)")
    if rom_count:
        parts.append(f"range in {rom_count} rep(s)")
    if elbow_count:
        parts.append(f"elbow position in {elbow_count} rep(s)")
    if upper_arm_count:
        parts.append(f"upper-arm stability in {upper_arm_count} rep(s)")
    return f"{rep_count} tricep extension rep(s) were analyzed. The main areas to clean up were " + ", ".join(parts) + "."


def _rep_breakdown(rep_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown = generic_rep_breakdown(rep_feedback)
    issue_details_map = {
        "bottom_position_shallow": (
            "Shallow start",
            "You did not bend enough before beginning the extension.",
        ),
        "lockout_incomplete": (
            "Incomplete lockout",
            "The elbow did not finish fully straight at the end of the rep.",
        ),
        "rom_incomplete": (
            "Shortened extension",
            "The total bend-to-lockout range was still shorter than expected.",
        ),
        "elbow_drift": (
            "Elbow drift",
            "Your elbow moved away from a stable path.",
        ),
        "upper_arm_instability": (
            "Upper arm unstable",
            "The upper arm shifted too much during the rep.",
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
