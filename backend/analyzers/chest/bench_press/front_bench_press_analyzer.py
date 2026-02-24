from typing import Any, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class FrontBenchPressAnalyzer(BaseAnalyzer):
    """
    Front view bench press analyzer:
      - Rep counting via elbow angle (ROM proxy)
      - Grip width ratio
      - Wrist height symmetry (bar tilt)
      - Elbow angle symmetry
      - Bar center drift (wrist midpoint x drift)
    """

    def __init__(self):
        super().__init__("bench_front")

    def analyze(
        self,
        videoPath: str,
        poseFrames: List[Any],
        videoMetadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = options or {}

        # ---------------------------
        # Rep thresholds (tuneable)
        # ---------------------------
        TOP_ANGLE = float(options.get("topAngleDeg", 155))
        BOTTOM_ANGLE = float(options.get("bottomAngleDeg", 95))
        HOLD_FRAMES = int(options.get("holdFrames", 4))

        # ---------------------------
        # Front-view form thresholds
        # ---------------------------
        GRIP_MIN = float(options.get("gripMinRatio", 1.0))
        GRIP_MAX = float(options.get("gripMaxRatio", 1.5))

        WRIST_Y_DIFF_WARN = float(options.get("wristYDiffWarn", 0.04))      # normalized y
        ELBOW_ANGLE_DIFF_WARN = float(options.get("elbowAngleDiffWarn", 15)) # degrees
        MID_X_DRIFT_WARN = float(options.get("midXDriftWarn", 0.06))         # normalized x drift per rep

        # ---------------------------
        # Helpers
        # ---------------------------
        def avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return (a + b) / 2.0

        def get_xy(lm: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
            p = lm.get(name)
            if p is None:
                return None
            return (float(p.x), float(p.y))

        # ---------------------------
        # State / outputs
        # ---------------------------
        repCount = 0
        state = "TOP"
        topHold = 0
        bottomHold = 0

        issues = []
        repFeedback = []

        # global aggregates
        grip_ratios: List[float] = []
        wrist_y_diffs: List[float] = []
        elbow_angle_diffs: List[float] = []

        # per-rep accumulators
        rep_total_frames = 0
        rep_bad_sym_frames = 0
        rep_mid_x_min = None
        rep_mid_x_max = None

        # ---------------------------
        # Main loop over frames
        # ---------------------------
        for pf in poseFrames:
            if not getattr(pf, "hasPose", False):
                continue

            lm = pf.landmarks

            ls = get_xy(lm, "left_shoulder")
            rs = get_xy(lm, "right_shoulder")
            lw = get_xy(lm, "left_wrist")
            rw = get_xy(lm, "right_wrist")

            if not (ls and rs and lw and rw):
                continue

            # grip ratio (x distance only)
            shoulder_dist = abs(ls[0] - rs[0]) + 1e-8
            wrist_dist = abs(lw[0] - rw[0])
            grip_ratio = wrist_dist / shoulder_dist
            grip_ratios.append(grip_ratio)

            # wrist symmetry (bar tilt proxy)
            wrist_y_diff = abs(lw[1] - rw[1])
            wrist_y_diffs.append(wrist_y_diff)

            # elbow angles
            left_elbow = getLandmarkAngle(lm, "left_shoulder", "left_elbow", "left_wrist")
            right_elbow = getLandmarkAngle(lm, "right_shoulder", "right_elbow", "right_wrist")
            elbow_angle = avg(left_elbow, right_elbow)
            if elbow_angle is None:
                continue

            if left_elbow is not None and right_elbow is not None:
                elbow_angle_diffs.append(abs(left_elbow - right_elbow))

            # bar midpoint drift (x)
            mid_x = (lw[0] + rw[0]) / 2.0

            # symmetry bad this frame?
            sym_bad = False
            if wrist_y_diff > WRIST_Y_DIFF_WARN:
                sym_bad = True
            if (left_elbow is not None and right_elbow is not None) and abs(left_elbow - right_elbow) > ELBOW_ANGLE_DIFF_WARN:
                sym_bad = True

            # ---------------------------
            # Rep state machine
            # ---------------------------
            if state == "TOP":
                if elbow_angle < BOTTOM_ANGLE:
                    bottomHold += 1
                    if bottomHold >= HOLD_FRAMES:
                        state = "DOWN"
                        bottomHold = 0
                        topHold = 0

                        # init per-rep accumulators
                        rep_total_frames = 0
                        rep_bad_sym_frames = 0
                        rep_mid_x_min = mid_x
                        rep_mid_x_max = mid_x
                else:
                    bottomHold = 0

            else:  # DOWN
                rep_total_frames += 1
                if sym_bad:
                    rep_bad_sym_frames += 1

                rep_mid_x_min = mid_x if rep_mid_x_min is None else min(rep_mid_x_min, mid_x)
                rep_mid_x_max = mid_x if rep_mid_x_max is None else max(rep_mid_x_max, mid_x)

                if elbow_angle > TOP_ANGLE:
                    topHold += 1
                    if topHold >= HOLD_FRAMES:
                        repCount += 1
                        state = "TOP"
                        topHold = 0
                        bottomHold = 0

                        bad_sym_ratio = (rep_bad_sym_frames / rep_total_frames) if rep_total_frames > 0 else 0.0
                        drift = (rep_mid_x_max - rep_mid_x_min) if (rep_mid_x_min is not None and rep_mid_x_max is not None) else 0.0

                        repFeedback.append({
                            "repIndex": repCount,
                            "romOk": True,
                            "symmetryOk": bad_sym_ratio < 0.25,
                            "symmetryBadRatio": round(bad_sym_ratio, 3),
                            "barCenteredOk": drift < MID_X_DRIFT_WARN,
                            "midXDrift": round(drift, 3),
                        })
                else:
                    topHold = 0

        # ---------------------------
        # Summaries -> issues
        # ---------------------------
        def safe_avg(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        grip_avg = safe_avg(grip_ratios)
        wrist_diff_avg = safe_avg(wrist_y_diffs)
        elbow_diff_avg = safe_avg(elbow_angle_diffs)

        if repCount == 0:
            issues.append(self.buildIssue(
                code="no_reps_detected",
                message="No reps detected (front). Try tuning angles or ensure shoulders/elbows/wrists are visible.",
                severity="medium"
            ))

        if grip_avg is not None and (grip_avg < GRIP_MIN or grip_avg > GRIP_MAX):
            issues.append(self.buildIssue(
                code="grip_width_out_of_range",
                message=f"Grip width ratio looks off (avg={grip_avg:.2f}). Aim for {GRIP_MIN:.1f}–{GRIP_MAX:.1f}× shoulder width.",
                severity="low"
            ))

        if wrist_diff_avg is not None and wrist_diff_avg > WRIST_Y_DIFF_WARN:
            issues.append(self.buildIssue(
                code="bar_tilt_or_asymmetry",
                message=f"Press looks uneven (avg wrist height diff={wrist_diff_avg:.3f}). Keep wrists level.",
                severity="low"
            ))

        if elbow_diff_avg is not None and elbow_diff_avg > ELBOW_ANGLE_DIFF_WARN:
            issues.append(self.buildIssue(
                code="elbow_asymmetry",
                message=f"Elbow angles differ (avg diff={elbow_diff_avg:.1f}°). Keep both arms moving together.",
                severity="low"
            ))

        # score: fraction of reps passing both symmetry + centering
        good_reps = sum(1 for r in repFeedback if r.get("symmetryOk") and r.get("barCenteredOk"))
        summaryScore = (good_reps / repCount) if repCount > 0 else None

        metrics = {
            "topAngleDeg": TOP_ANGLE,
            "bottomAngleDeg": BOTTOM_ANGLE,
            "holdFrames": HOLD_FRAMES,
            "gripMinRatio": GRIP_MIN,
            "gripMaxRatio": GRIP_MAX,
            "wristYDiffWarn": WRIST_Y_DIFF_WARN,
            "elbowAngleDiffWarn": ELBOW_ANGLE_DIFF_WARN,
            "midXDriftWarn": MID_X_DRIFT_WARN,
            "gripRatioAvg": None if grip_avg is None else round(grip_avg, 3),
            "wristYDiffAvg": None if wrist_diff_avg is None else round(wrist_diff_avg, 4),
            "elbowAngleDiffAvg": None if elbow_diff_avg is None else round(elbow_diff_avg, 2),
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=[],
            message="Front bench press analysis completed."
        )