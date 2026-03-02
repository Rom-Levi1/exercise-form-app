from collections import deque
from typing import Any, Deque, Dict, List, Optional

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class PullUpAnalyzer(BaseAnalyzer):
    """
    Back-view pull-up analyzer (TOP-crossing rep count):
      - Counts a rep when athlete ENTERS TOP zone (no dead-hang required)
      - TOP detection: armpit angle if available, otherwise elbow angle fallback
      - Symmetry (requested rule):
            Compare LEFT vs RIGHT wrist-shoulder X *range* during the TOP phase
            => similarity score in [0..1] (1 = identical ranges)
      - Optional angle symmetry info (elbow/armpit diff snapshot)

    Removed:
      - wrist Y level check (not useful for pull-ups)
    """

    def __init__(self):
        super().__init__("pullup_back")

    def analyze(
        self,
        videoPath: str,
        poseFrames: List[Any],
        videoMetadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = options or {}

        # ---------------------------
        # Thresholds
        # ---------------------------
        TOP_ELBOW_ANGLE = float(options.get("topElbowAngleDeg", 95))
        TOP_ARMPIT_ANGLE = float(options.get("topArmpitAngleDeg", 105))

        # Must leave top before counting another rep
        LEAVE_TOP_ELBOW = float(options.get("leaveTopElbowDeg", 120))

        # Hysteresis + smoothing
        HYST_DEG = float(options.get("hysteresisDeg", 6))
        SMOOTH_WINDOW = int(options.get("smoothWindow", 5))

        # Height rule
        REQUIRE_ARMPIT_FOR_HEIGHT = bool(options.get("requireArmpitForHeight", False))

        # Optional angle-based symmetry info
        ENABLE_ANGLE_SYMMETRY = bool(options.get("enableAngleSymmetry", True))
        SYM_ELBOW_DIFF_WARN = float(options.get("symElbowDiffWarnDeg", 18))
        SYM_ARMPIT_DIFF_WARN = float(options.get("symArmpitDiffWarnDeg", 18))

        # ---------------------------
        # Helpers
        # ---------------------------
        def safe_avg(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        def avg2(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return (a + b) / 2.0

        def range_of(xs: List[float]) -> Optional[float]:
            if len(xs) < 2:
                return None
            return max(xs) - min(xs)

        def range_similarity(r1: Optional[float], r2: Optional[float]) -> Optional[float]:
            if r1 is None or r2 is None:
                return None
            eps = 1e-6
            return 1.0 - abs(r1 - r2) / (r1 + r2 + eps)

        # ---------------------------
        # Outputs
        # ---------------------------
        repCount = 0
        issues = []
        repFeedback: List[Dict[str, Any]] = []

        sym_elbow_diffs: List[float] = []
        sym_armpit_diffs: List[float] = []

        elbow_buf: Deque[float] = deque(maxlen=max(1, SMOOTH_WINDOW))

        in_top = False
        pending_rep: Optional[Dict[str, Any]] = None
        dxL_samples: List[float] = []
        dxR_samples: List[float] = []

        # Debug metrics
        skipped_missing_elbow = 0
        valid_elbow_frames = 0
        elbow_min = 999.0
        elbow_max = -999.0
        top_hit_frames = 0
        leave_top_frames = 0
        armpit_available_frames = 0

        # ---------------------------
        # Main loop
        # ---------------------------
        for pf in poseFrames:
            if not getattr(pf, "hasPose", False):
                continue

            lm = pf.landmarks

            # Elbow angles
            left_elbow = getLandmarkAngle(lm, "left_shoulder", "left_elbow", "left_wrist")
            right_elbow = getLandmarkAngle(lm, "right_shoulder", "right_elbow", "right_wrist")
            elbow_angle = avg2(left_elbow, right_elbow)

            if elbow_angle is None:
                skipped_missing_elbow += 1
                continue

            elbow_buf.append(float(elbow_angle))
            elbow_s = sum(elbow_buf) / len(elbow_buf)

            valid_elbow_frames += 1
            elbow_min = min(elbow_min, elbow_s)
            elbow_max = max(elbow_max, elbow_s)

            # Armpit angle (optional)
            left_armpit = getLandmarkAngle(lm, "left_elbow", "left_shoulder", "left_hip")
            right_armpit = getLandmarkAngle(lm, "right_elbow", "right_shoulder", "right_hip")
            armpit_angle = avg2(left_armpit, right_armpit)
            if armpit_angle is not None:
                armpit_available_frames += 1

            # dx samples (wrist relative to shoulder)
            lw = lm.get("left_wrist")
            rw = lm.get("right_wrist")
            ls = lm.get("left_shoulder")
            rs = lm.get("right_shoulder")

            dxL = dxR = None
            if lw is not None and ls is not None:
                dxL = float(lw.x) - float(ls.x)
            if rw is not None and rs is not None:
                dxR = float(rw.x) - float(rs.x)

            # Optional angle symmetry snapshot
            angle_sym_bad = False
            if ENABLE_ANGLE_SYMMETRY and left_elbow is not None and right_elbow is not None:
                d = abs(left_elbow - right_elbow)
                sym_elbow_diffs.append(d)
                if d > SYM_ELBOW_DIFF_WARN:
                    angle_sym_bad = True

            if ENABLE_ANGLE_SYMMETRY and left_armpit is not None and right_armpit is not None:
                d = abs(left_armpit - right_armpit)
                sym_armpit_diffs.append(d)
                if d > SYM_ARMPIT_DIFF_WARN:
                    angle_sym_bad = True

            # ---------------------------
            # TOP detection
            # ---------------------------
            elbow_ok = elbow_s <= (TOP_ELBOW_ANGLE + HYST_DEG)
            armpit_ok = (armpit_angle is not None) and (float(armpit_angle) <= (TOP_ARMPIT_ANGLE + HYST_DEG))
            top_hit = armpit_ok or elbow_ok
            if top_hit:
                top_hit_frames += 1

            left_top = elbow_s >= (LEAVE_TOP_ELBOW - HYST_DEG)
            if left_top:
                leave_top_frames += 1

            # ---------------------------
            # Rep counting: ENTER TOP -> start pending rep
            # finalize rep on LEAVE TOP (for range similarity)
            # ---------------------------
            if not in_top:
                if top_hit:
                    repCount += 1
                    in_top = True

                    dxL_samples = []
                    dxR_samples = []
                    if dxL is not None:
                        dxL_samples.append(dxL)
                    if dxR is not None:
                        dxR_samples.append(dxR)

                    # Height ok
                    if REQUIRE_ARMPIT_FOR_HEIGHT:
                        height_ok = bool(armpit_ok)
                    else:
                        height_ok = bool(armpit_ok or elbow_ok)

                    pending_rep = {
                        "repIndex": repCount,
                        "heightOk": bool(height_ok),
                        "elbowDegAtTop": round(elbow_s, 1),
                        "armpitDegAtTop": None if armpit_angle is None else round(float(armpit_angle), 1),
                        "usedArmpitForTop": bool(armpit_ok),
                        "angleSymmetryBadAtTop": bool(angle_sym_bad),
                    }

            else:
                if dxL is not None:
                    dxL_samples.append(dxL)
                if dxR is not None:
                    dxR_samples.append(dxR)

                if left_top:
                    in_top = False

                    rangeL = range_of(dxL_samples)
                    rangeR = range_of(dxR_samples)
                    sim = range_similarity(rangeL, rangeR)

                    # neutral if missing data
                    symmetry_score = 1.0 if sim is None else max(0.0, min(1.0, sim))

                    if pending_rep is not None:
                        pending_rep["wristShoulderRangeLeft"] = None if rangeL is None else round(rangeL, 4)
                        pending_rep["wristShoulderRangeRight"] = None if rangeR is None else round(rangeR, 4)
                        pending_rep["wristShoulderRangeSimilarity"] = None if sim is None else round(sim, 3)
                        pending_rep["symmetryScore"] = round(symmetry_score, 3)

                        repFeedback.append(pending_rep)

                    pending_rep = None
                    dxL_samples = []
                    dxR_samples = []

        # ---------------------------
        # Summaries -> issues
        # ---------------------------
        sym_elbow_avg = safe_avg(sym_elbow_diffs)
        sym_armpit_avg = safe_avg(sym_armpit_diffs)

        if repCount == 0:
            issues.append(
                self.buildIssue(
                    code="no_reps_detected",
                    message="No pull-up reps detected. Try loosening TOP thresholds or check pose quality.",
                    severity="medium",
                )
            )

        if ENABLE_ANGLE_SYMMETRY and sym_elbow_avg is not None and sym_elbow_avg > SYM_ELBOW_DIFF_WARN:
            issues.append(
                self.buildIssue(
                    code="elbow_asymmetry",
                    message=f"Elbow angles differ often (avg diff={sym_elbow_avg:.1f}°).",
                    severity="low",
                )
            )

        if ENABLE_ANGLE_SYMMETRY and sym_armpit_avg is not None and sym_armpit_avg > SYM_ARMPIT_DIFF_WARN:
            issues.append(
                self.buildIssue(
                    code="armpit_asymmetry",
                    message=f"Armpit angles differ often (avg diff={sym_armpit_avg:.1f}°).",
                    severity="low",
                )
            )

        # ---------------------------
        # Scoring (continuous)
        # ---------------------------
        # RepScore = heightOk * symmetryScore
        rep_scores: List[float] = []
        for r in repFeedback:
            height = 1.0 if r.get("heightOk") else 0.0
            sym = float(r.get("symmetryScore", 1.0))
            rep_scores.append(height * sym)

        summaryScore = (sum(rep_scores) / len(rep_scores)) if rep_scores else None

        metrics = {
            "topElbowAngleDeg": TOP_ELBOW_ANGLE,
            "topArmpitAngleDeg": TOP_ARMPIT_ANGLE,
            "leaveTopElbowDeg": LEAVE_TOP_ELBOW,
            "hysteresisDeg": HYST_DEG,
            "smoothWindow": SMOOTH_WINDOW,
            "requireArmpitForHeight": REQUIRE_ARMPIT_FOR_HEIGHT,
            "enableAngleSymmetry": ENABLE_ANGLE_SYMMETRY,
            "symElbowDiffWarnDeg": SYM_ELBOW_DIFF_WARN,
            "symArmpitDiffWarnDeg": SYM_ARMPIT_DIFF_WARN,
            "skippedMissingElbowFrames": skipped_missing_elbow,
            "validElbowFrames": valid_elbow_frames,
            "elbowAngleMin": None if valid_elbow_frames == 0 else round(elbow_min, 1),
            "elbowAngleMax": None if valid_elbow_frames == 0 else round(elbow_max, 1),
            "topHitFrames": top_hit_frames,
            "leaveTopFrames": leave_top_frames,
            "armpitAvailableFrames": armpit_available_frames,
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=[],
            message="Back pull-up analysis completed.",
        )