from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class SideBenchPressAnalyzer(BaseAnalyzer):
    """
    Side view bench press analyzer (side-preferred checks):
      - Rep counting via elbow angle with hysteresis + smoothing (robust)
      - ROM validation per rep (min/max elbow angle + min ROM)
      - Bar path proxy (side): horizontal drift of wrist relative to shoulder during rep
      - Wrist-over-elbow stacking proxy: |wrist.x - elbow.x|
      - Optional elbow tuck proxy: angle at shoulder (elbow relative to torso line shoulder->hip)
    """

    def __init__(self):
        super().__init__("bench_side")

    def analyze(
        self,
        videoPath: str,
        poseFrames: List[Any],
        videoMetadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = options or {}

        # ---------------------------
        # Which side is visible?
        # ---------------------------
        side = str(options.get("side", "left")).lower()
        if side not in ("left", "right"):
            side = "left"

        SHOULDER = f"{side}_shoulder"
        ELBOW = f"{side}_elbow"
        WRIST = f"{side}_wrist"
        HIP = f"{side}_hip"

        # ---------------------------
        # Rep thresholds (tuneable)
        # ---------------------------
        TOP_ANGLE = float(options.get("topAngleDeg", 155))
        BOTTOM_ANGLE = float(options.get("bottomAngleDeg", 105))

        # Hysteresis band (prevents flicker around thresholds)
        HYST_DEG = float(options.get("hysteresisDeg", 5))

        # Light smoothing window for elbow angle (0 disables)
        SMOOTH_WINDOW = int(options.get("smoothWindow", 5))

        # ---------------------------
        # ROM validation thresholds
        # ---------------------------
        MIN_ROM_DEG = float(options.get("minRomDeg", 50))
        BOTTOM_MARGIN = float(options.get("bottomMarginDeg", 5))
        TOP_MARGIN = float(options.get("topMarginDeg", 5))

        # ---------------------------
        # Side-view form thresholds
        # ---------------------------
        WRIST_ELBOW_X_WARN = float(options.get("wristElbowXWarn", 0.04))
        BAR_REL_X_DRIFT_WARN = float(options.get("barRelXDriftWarn", 0.06))

        ENABLE_TUCK_CHECK = bool(options.get("enableTuckCheck", False))
        TUCK_MIN_DEG = float(options.get("tuckMinDeg", 20))
        TUCK_MAX_DEG = float(options.get("tuckMaxDeg", 70))

        # ---------------------------
        # Helpers
        # ---------------------------
        def get_xy(lm: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
            p = lm.get(name)
            if p is None:
                return None
            return (float(p.x), float(p.y))

        def safe_avg(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        def in_top_zone(a: float) -> bool:
            return a >= (TOP_ANGLE - HYST_DEG)

        def in_bottom_zone(a: float) -> bool:
            return a <= (BOTTOM_ANGLE + HYST_DEG)

        # ---------------------------
        # State / outputs
        # ---------------------------
        repCount = 0
        issues = []
        repFeedback = []

        # Global aggregates
        wrist_elbow_x_diffs: List[float] = []
        rep_bar_rel_x_drifts: List[float] = []
        shoulder_tuck_angles: List[float] = []

        # Smoothing buffer
        angle_buf: Deque[float] = deque(maxlen=max(1, SMOOTH_WINDOW))

        # Rep state:
        #   "UNKNOWN" -> "TOP_READY" or "DOWN_IN_REP"
        state = "UNKNOWN"

        # Per-rep accumulators
        rep_total_frames = 0
        rep_bad_stack_frames = 0

        rep_bar_rel_x_min = None
        rep_bar_rel_x_max = None

        rep_elbow_min = None
        rep_elbow_max = None

        rep_tuck_min = None
        rep_tuck_max = None
        rep_start_frame_index = None

        # ---------------------------
        # Main loop
        # ---------------------------
        for pf in poseFrames:
            if not getattr(pf, "hasPose", False):
                continue

            lm = pf.landmarks
            sh = get_xy(lm, SHOULDER)
            el = get_xy(lm, ELBOW)
            wr = get_xy(lm, WRIST)
            hp = get_xy(lm, HIP) if ENABLE_TUCK_CHECK else None

            if not (sh and el and wr):
                continue

            raw_angle = getLandmarkAngle(lm, SHOULDER, ELBOW, WRIST)
            if raw_angle is None:
                continue

            # Smooth angle
            angle_buf.append(float(raw_angle))
            elbow_angle = sum(angle_buf) / len(angle_buf)

            # Initialize state on first valid frame
            if state == "UNKNOWN":
                state = "TOP_READY" if in_top_zone(elbow_angle) else "DOWN_IN_REP"
                # If we start mid-rep, initialize per-rep trackers now
                if state == "DOWN_IN_REP":
                    rep_total_frames = 0
                    rep_bad_stack_frames = 0
                    bar_rel_x = wr[0] - sh[0]
                    rep_bar_rel_x_min = bar_rel_x
                    rep_bar_rel_x_max = bar_rel_x
                    rep_elbow_min = elbow_angle
                    rep_elbow_max = elbow_angle
                    rep_tuck_min = float(tuck_angle) if tuck_angle is not None else None
                    rep_tuck_max = float(tuck_angle) if tuck_angle is not None else None
                    rep_start_frame_index = getattr(pf, "frameIndex", None)

            # stacking proxy
            wrist_elbow_x_diff = abs(wr[0] - el[0])
            wrist_elbow_x_diffs.append(wrist_elbow_x_diff)
            stack_bad = wrist_elbow_x_diff > WRIST_ELBOW_X_WARN

            # bar path proxy (relative to shoulder)
            bar_rel_x = wr[0] - sh[0]

            # tuck proxy (optional)
            tuck_angle = None
            if ENABLE_TUCK_CHECK and hp is not None:
                tuck_angle = getLandmarkAngle(lm, ELBOW, SHOULDER, HIP)
                if tuck_angle is not None:
                    shoulder_tuck_angles.append(float(tuck_angle))

            # ---------------------------
            # Rep logic with hysteresis
            # ---------------------------
            if state == "TOP_READY":
                # Wait until we truly reach bottom zone to start a rep
                if in_bottom_zone(elbow_angle):
                    state = "DOWN_IN_REP"

                    rep_total_frames = 0
                    rep_bad_stack_frames = 0

                    rep_bar_rel_x_min = bar_rel_x
                    rep_bar_rel_x_max = bar_rel_x

                    rep_elbow_min = elbow_angle
                    rep_elbow_max = elbow_angle

                    rep_tuck_min = float(tuck_angle) if tuck_angle is not None else None
                    rep_tuck_max = float(tuck_angle) if tuck_angle is not None else None
                    rep_start_frame_index = getattr(pf, "frameIndex", None)

            else:  # DOWN_IN_REP
                rep_total_frames += 1
                if stack_bad:
                    rep_bad_stack_frames += 1

                rep_bar_rel_x_min = bar_rel_x if rep_bar_rel_x_min is None else min(rep_bar_rel_x_min, bar_rel_x)
                rep_bar_rel_x_max = bar_rel_x if rep_bar_rel_x_max is None else max(rep_bar_rel_x_max, bar_rel_x)

                rep_elbow_min = elbow_angle if rep_elbow_min is None else min(rep_elbow_min, elbow_angle)
                rep_elbow_max = elbow_angle if rep_elbow_max is None else max(rep_elbow_max, elbow_angle)

                if ENABLE_TUCK_CHECK and tuck_angle is not None:
                    t = float(tuck_angle)
                    rep_tuck_min = t if rep_tuck_min is None else min(rep_tuck_min, t)
                    rep_tuck_max = t if rep_tuck_max is None else max(rep_tuck_max, t)

                # Rep completes when we return to TOP zone
                if in_top_zone(elbow_angle):
                    repCount += 1
                    state = "TOP_READY"

                    # --- stacking
                    stack_bad_ratio = (rep_bad_stack_frames / rep_total_frames) if rep_total_frames > 0 else 0.0
                    stacking_ok = stack_bad_ratio < 0.25

                    # --- bar path drift
                    bar_rel_x_drift = (
                        (rep_bar_rel_x_max - rep_bar_rel_x_min)
                        if (rep_bar_rel_x_min is not None and rep_bar_rel_x_max is not None)
                        else 0.0
                    )
                    rep_bar_rel_x_drifts.append(bar_rel_x_drift)
                    bar_path_ok = bar_rel_x_drift < BAR_REL_X_DRIFT_WARN

                    # --- ROM
                    rom = (
                        (rep_elbow_max - rep_elbow_min)
                        if (rep_elbow_min is not None and rep_elbow_max is not None)
                        else 0.0
                    )
                    hit_bottom = (rep_elbow_min is not None) and (rep_elbow_min <= (BOTTOM_ANGLE + BOTTOM_MARGIN))
                    hit_top = (rep_elbow_max is not None) and (rep_elbow_max >= (TOP_ANGLE - TOP_MARGIN))
                    rom_ok = hit_bottom and hit_top and (rom >= MIN_ROM_DEG)

                    # --- tuck (optional)
                    tuck_ok = None
                    if ENABLE_TUCK_CHECK and (rep_tuck_min is not None or rep_tuck_max is not None):
                        tuck_avg = (
                            (rep_tuck_min + rep_tuck_max) / 2.0
                            if (rep_tuck_min is not None and rep_tuck_max is not None)
                            else (rep_tuck_min if rep_tuck_min is not None else rep_tuck_max)
                        )
                        tuck_ok = (tuck_avg is not None) and (TUCK_MIN_DEG <= tuck_avg <= TUCK_MAX_DEG)

                    rep_issue_codes = []
                    quality_penalty = 0.0
                    if not rom_ok:
                        rep_issue_codes.append("rom_incomplete")
                        quality_penalty += 40.0
                    if not bar_path_ok:
                        rep_issue_codes.append("bar_path_drift")
                        quality_penalty += 30.0
                    if not stacking_ok:
                        rep_issue_codes.append("wrist_elbow_stacking")
                        quality_penalty += 30.0
                    if ENABLE_TUCK_CHECK and tuck_ok is False:
                        rep_issue_codes.append("elbow_tuck_off")
                        quality_penalty += 15.0

                    rep_quality = max(0.0, 100.0 - quality_penalty)

                    repFeedback.append(
                        {
                            "repIndex": repCount,
                            "side": side,
                            "startFrameIndex": rep_start_frame_index,
                            "endFrameIndex": getattr(pf, "frameIndex", None),
                            "quality": round(rep_quality, 1),
                            "issues": rep_issue_codes,

                            "romOk": rom_ok,
                            "romDeg": round(rom, 1),
                            "minElbowDeg": None if rep_elbow_min is None else round(rep_elbow_min, 1),
                            "maxElbowDeg": None if rep_elbow_max is None else round(rep_elbow_max, 1),
                            "hitBottom": hit_bottom,
                            "hitTop": hit_top,

                            "barPathOk": bar_path_ok,
                            "barRelXDrift": round(bar_rel_x_drift, 3),

                            "stackingOk": stacking_ok,
                            "stackingBadRatio": round(stack_bad_ratio, 3),

                            "tuckOk": tuck_ok,
                            "tuckAngleMinDeg": None if rep_tuck_min is None else round(rep_tuck_min, 1),
                            "tuckAngleMaxDeg": None if rep_tuck_max is None else round(rep_tuck_max, 1),
                        }
                    )

                    # Reset per-rep trackers
                    rep_total_frames = 0
                    rep_bad_stack_frames = 0
                    rep_bar_rel_x_min = None
                    rep_bar_rel_x_max = None
                    rep_elbow_min = None
                    rep_elbow_max = None
                    rep_tuck_min = None
                    rep_tuck_max = None
                    rep_start_frame_index = None

        # ---------------------------
        # Summaries -> issues
        # ---------------------------
        wrist_elbow_x_avg = safe_avg(wrist_elbow_x_diffs)
        bar_rel_x_drift_avg = safe_avg(rep_bar_rel_x_drifts)
        tuck_angle_avg = safe_avg(shoulder_tuck_angles) if ENABLE_TUCK_CHECK else None

        if repCount == 0:
            issues.append(
                self.buildIssue(
                    code="no_reps_detected",
                    message="No reps detected (side). Try tuning angles or ensure shoulder/elbow/wrist are visible.",
                    severity="medium",
                )
            )

        if wrist_elbow_x_avg is not None and wrist_elbow_x_avg > WRIST_ELBOW_X_WARN:
            issues.append(
                self.buildIssue(
                    code="wrist_not_stacked_over_elbow",
                    message=f"Wrist often not stacked over elbow (avg |wrist.x-elbow.x|={wrist_elbow_x_avg:.3f}).",
                    severity="low",
                )
            )

        if bar_rel_x_drift_avg is not None and bar_rel_x_drift_avg > BAR_REL_X_DRIFT_WARN:
            issues.append(
                self.buildIssue(
                    code="bar_path_horizontal_drift",
                    message=f"Bar path drifts horizontally (avg wrist.x-shoulder.x drift per rep={bar_rel_x_drift_avg:.3f}).",
                    severity="low",
                )
            )

        if ENABLE_TUCK_CHECK and tuck_angle_avg is not None and not (TUCK_MIN_DEG <= tuck_angle_avg <= TUCK_MAX_DEG):
            issues.append(
                self.buildIssue(
                    code="elbow_tuck_out_of_range",
                    message=f"Elbow tuck angle looks off (avg={tuck_angle_avg:.1f}°). Target {TUCK_MIN_DEG:.0f}–{TUCK_MAX_DEG:.0f}° (heuristic).",
                    severity="low",
                )
            )

        # Score: reps passing core checks (+ tuck if enabled)
        def is_good_rep(r: Dict[str, Any]) -> bool:
            if not (r.get("romOk") and r.get("barPathOk") and r.get("stackingOk")):
                return False
            if ENABLE_TUCK_CHECK:
                return (r.get("tuckOk") is None) or bool(r.get("tuckOk"))
            return True

        good_reps = sum(1 for r in repFeedback if is_good_rep(r))
        summaryScore = (good_reps / repCount) if repCount > 0 else None

        metrics = {
            "side": side,

            "topAngleDeg": TOP_ANGLE,
            "bottomAngleDeg": BOTTOM_ANGLE,
            "hysteresisDeg": HYST_DEG,
            "smoothWindow": SMOOTH_WINDOW,

            "minRomDeg": MIN_ROM_DEG,
            "bottomMarginDeg": BOTTOM_MARGIN,
            "topMarginDeg": TOP_MARGIN,

            "wristElbowXWarn": WRIST_ELBOW_X_WARN,
            "barRelXDriftWarn": BAR_REL_X_DRIFT_WARN,

            "enableTuckCheck": ENABLE_TUCK_CHECK,
            "tuckMinDeg": TUCK_MIN_DEG if ENABLE_TUCK_CHECK else None,
            "tuckMaxDeg": TUCK_MAX_DEG if ENABLE_TUCK_CHECK else None,

            "wristElbowXAvg": None if wrist_elbow_x_avg is None else round(wrist_elbow_x_avg, 4),
            "barRelXDriftAvg": None if bar_rel_x_drift_avg is None else round(bar_rel_x_drift_avg, 4),
            "tuckAngleAvg": None if tuck_angle_avg is None else round(tuck_angle_avg, 2),
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=[],
            message="Side bench press analysis completed.",
        )
