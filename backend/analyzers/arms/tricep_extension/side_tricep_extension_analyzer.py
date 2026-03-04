from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class SideTricepExtensionAnalyzer(BaseAnalyzer):
    """
    Side view tricep extension analyzer:
      - Rep counting via elbow angle with hysteresis + smoothing
      - ROM validation per rep (min/max elbow angle + minimum ROM)
      - Upper-arm stability proxy (elbow position drift relative to shoulder)
      - Shoulder angle stability proxy (upper-arm orientation drift)
    """

    def __init__(self):
        super().__init__("tricep_extension_side")

    def analyze(
        self,
        videoPath: str,
        poseFrames: List[Any],
        videoMetadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = options or {}

        side = str(options.get("side", "left")).lower()
        if side not in ("left", "right"):
            side = "left"

        SHOULDER = f"{side}_shoulder"
        ELBOW = f"{side}_elbow"
        WRIST = f"{side}_wrist"
        HIP = f"{side}_hip"

        # Rep thresholds
        TOP_ANGLE = float(options.get("topAngleDeg", 160))
        BOTTOM_ANGLE = float(options.get("bottomAngleDeg", 85))
        HYST_DEG = float(options.get("hysteresisDeg", 5))
        SMOOTH_WINDOW = int(options.get("smoothWindow", 5))

        # ROM thresholds
        MIN_ROM_DEG = float(options.get("minRomDeg", 65))
        BOTTOM_MARGIN = float(options.get("bottomMarginDeg", 5))
        TOP_MARGIN = float(options.get("topMarginDeg", 5))

        # Stability thresholds
        ELBOW_REL_X_DRIFT_WARN = float(options.get("elbowRelXDriftWarn", 0.05))
        UPPER_ARM_ANGLE_DRIFT_WARN = float(options.get("upperArmAngleDriftWarn", 20))

        def get_xy(lm: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
            p = lm.get(name)
            if p is None:
                return None
            return (float(p.x), float(p.y))

        def safe_avg(values: List[float]) -> Optional[float]:
            return (sum(values) / len(values)) if values else None

        def in_top_zone(angle: float) -> bool:
            return angle >= (TOP_ANGLE - HYST_DEG)

        def in_bottom_zone(angle: float) -> bool:
            return angle <= (BOTTOM_ANGLE + HYST_DEG)

        repCount = 0
        issues: List[Dict[str, Any]] = []
        repFeedback: List[Dict[str, Any]] = []
        warnings: List[str] = []

        angleBuf: Deque[float] = deque(maxlen=max(1, SMOOTH_WINDOW))
        state = "UNKNOWN"  # "TOP_READY" | "DOWN_IN_REP"

        # Global aggregates
        rep_elbow_rel_x_drifts: List[float] = []
        rep_upper_arm_angle_drifts: List[float] = []

        # Per-rep accumulators
        rep_elbow_min: Optional[float] = None
        rep_elbow_max: Optional[float] = None
        rep_elbow_rel_x_min: Optional[float] = None
        rep_elbow_rel_x_max: Optional[float] = None
        rep_upper_arm_angle_min: Optional[float] = None
        rep_upper_arm_angle_max: Optional[float] = None

        for pf in poseFrames:
            if not getattr(pf, "hasPose", False):
                continue

            lm = pf.landmarks
            sh = get_xy(lm, SHOULDER)
            el = get_xy(lm, ELBOW)
            wr = get_xy(lm, WRIST)
            hp = get_xy(lm, HIP)

            if not (sh and el and wr):
                continue

            rawElbowAngle = getLandmarkAngle(lm, SHOULDER, ELBOW, WRIST)
            if rawElbowAngle is None:
                continue

            angleBuf.append(float(rawElbowAngle))
            elbowAngle = sum(angleBuf) / len(angleBuf)

            # Approximate upper-arm angle at shoulder (elbow relative to torso line).
            upperArmAngle = getLandmarkAngle(lm, ELBOW, SHOULDER, HIP) if hp else None

            elbowRelX = el[0] - sh[0]

            if state == "UNKNOWN":
                state = "TOP_READY" if in_top_zone(elbowAngle) else "DOWN_IN_REP"
                if state == "DOWN_IN_REP":
                    rep_elbow_min = elbowAngle
                    rep_elbow_max = elbowAngle
                    rep_elbow_rel_x_min = elbowRelX
                    rep_elbow_rel_x_max = elbowRelX
                    if upperArmAngle is not None:
                        rep_upper_arm_angle_min = float(upperArmAngle)
                        rep_upper_arm_angle_max = float(upperArmAngle)

            if state == "TOP_READY":
                if in_bottom_zone(elbowAngle):
                    state = "DOWN_IN_REP"
                    rep_elbow_min = elbowAngle
                    rep_elbow_max = elbowAngle
                    rep_elbow_rel_x_min = elbowRelX
                    rep_elbow_rel_x_max = elbowRelX
                    rep_upper_arm_angle_min = (
                        float(upperArmAngle) if upperArmAngle is not None else None
                    )
                    rep_upper_arm_angle_max = (
                        float(upperArmAngle) if upperArmAngle is not None else None
                    )
                continue

            # state == DOWN_IN_REP
            rep_elbow_min = elbowAngle if rep_elbow_min is None else min(rep_elbow_min, elbowAngle)
            rep_elbow_max = elbowAngle if rep_elbow_max is None else max(rep_elbow_max, elbowAngle)
            rep_elbow_rel_x_min = (
                elbowRelX if rep_elbow_rel_x_min is None else min(rep_elbow_rel_x_min, elbowRelX)
            )
            rep_elbow_rel_x_max = (
                elbowRelX if rep_elbow_rel_x_max is None else max(rep_elbow_rel_x_max, elbowRelX)
            )

            if upperArmAngle is not None:
                rep_upper_arm_angle_min = (
                    float(upperArmAngle)
                    if rep_upper_arm_angle_min is None
                    else min(rep_upper_arm_angle_min, float(upperArmAngle))
                )
                rep_upper_arm_angle_max = (
                    float(upperArmAngle)
                    if rep_upper_arm_angle_max is None
                    else max(rep_upper_arm_angle_max, float(upperArmAngle))
                )

            # Count rep when returning to top after entering bottom zone.
            if in_top_zone(elbowAngle):
                repCount += 1

                rom = (
                    (rep_elbow_max - rep_elbow_min)
                    if (rep_elbow_min is not None and rep_elbow_max is not None)
                    else 0.0
                )
                hitBottom = (rep_elbow_min is not None) and (
                    rep_elbow_min <= (BOTTOM_ANGLE + BOTTOM_MARGIN)
                )
                hitTop = (rep_elbow_max is not None) and (rep_elbow_max >= (TOP_ANGLE - TOP_MARGIN))
                romOk = hitBottom and hitTop and (rom >= MIN_ROM_DEG)

                elbowRelXDrift = (
                    (rep_elbow_rel_x_max - rep_elbow_rel_x_min)
                    if (rep_elbow_rel_x_min is not None and rep_elbow_rel_x_max is not None)
                    else 0.0
                )
                elbowStable = elbowRelXDrift < ELBOW_REL_X_DRIFT_WARN

                upperArmAngleDrift = (
                    (rep_upper_arm_angle_max - rep_upper_arm_angle_min)
                    if (
                        rep_upper_arm_angle_min is not None
                        and rep_upper_arm_angle_max is not None
                    )
                    else None
                )
                upperArmStable = (
                    None
                    if upperArmAngleDrift is None
                    else (upperArmAngleDrift < UPPER_ARM_ANGLE_DRIFT_WARN)
                )

                rep_elbow_rel_x_drifts.append(elbowRelXDrift)
                if upperArmAngleDrift is not None:
                    rep_upper_arm_angle_drifts.append(upperArmAngleDrift)

                repFeedback.append(
                    {
                        "repIndex": repCount,
                        "side": side,
                        "romOk": romOk,
                        "romDeg": round(rom, 1),
                        "minElbowDeg": None if rep_elbow_min is None else round(rep_elbow_min, 1),
                        "maxElbowDeg": None if rep_elbow_max is None else round(rep_elbow_max, 1),
                        "hitBottom": hitBottom,
                        "hitTop": hitTop,
                        "elbowStable": elbowStable,
                        "elbowRelXDrift": round(elbowRelXDrift, 3),
                        "upperArmStable": upperArmStable,
                        "upperArmAngleDriftDeg": (
                            None if upperArmAngleDrift is None else round(upperArmAngleDrift, 1)
                        ),
                    }
                )

                # Prepare for the next rep.
                state = "TOP_READY"
                rep_elbow_min = None
                rep_elbow_max = None
                rep_elbow_rel_x_min = None
                rep_elbow_rel_x_max = None
                rep_upper_arm_angle_min = None
                rep_upper_arm_angle_max = None

        elbowRelXDriftAvg = safe_avg(rep_elbow_rel_x_drifts)
        upperArmAngleDriftAvg = safe_avg(rep_upper_arm_angle_drifts)

        if repCount == 0:
            issues.append(
                self.buildIssue(
                    code="no_reps_detected",
                    message=(
                        "No side tricep extension reps detected. Ensure shoulder/elbow/wrist are "
                        "visible and tune thresholds if needed."
                    ),
                    severity="medium",
                )
            )

        if elbowRelXDriftAvg is not None and elbowRelXDriftAvg > ELBOW_REL_X_DRIFT_WARN:
            issues.append(
                self.buildIssue(
                    code="elbow_drift",
                    message=(
                        f"Elbow drift relative to shoulder looked high (avg drift={elbowRelXDriftAvg:.3f})."
                    ),
                    severity="low",
                )
            )

        if (
            upperArmAngleDriftAvg is not None
            and upperArmAngleDriftAvg > UPPER_ARM_ANGLE_DRIFT_WARN
        ):
            issues.append(
                self.buildIssue(
                    code="upper_arm_instability",
                    message=(
                        "Upper-arm angle changed a lot during reps "
                        f"(avg drift={upperArmAngleDriftAvg:.1f} deg)."
                    ),
                    severity="low",
                )
            )

        def isGoodRep(rep: Dict[str, Any]) -> bool:
            if not rep.get("romOk"):
                return False
            if not rep.get("elbowStable"):
                return False
            if rep.get("upperArmStable") is False:
                return False
            return True

        goodReps = sum(1 for rep in repFeedback if isGoodRep(rep))
        summaryScore = (goodReps / repCount) if repCount > 0 else None

        metrics = {
            "side": side,
            "topAngleDeg": TOP_ANGLE,
            "bottomAngleDeg": BOTTOM_ANGLE,
            "hysteresisDeg": HYST_DEG,
            "smoothWindow": SMOOTH_WINDOW,
            "minRomDeg": MIN_ROM_DEG,
            "bottomMarginDeg": BOTTOM_MARGIN,
            "topMarginDeg": TOP_MARGIN,
            "elbowRelXDriftWarn": ELBOW_REL_X_DRIFT_WARN,
            "upperArmAngleDriftWarn": UPPER_ARM_ANGLE_DRIFT_WARN,
            "elbowRelXDriftAvg": (
                None if elbowRelXDriftAvg is None else round(elbowRelXDriftAvg, 4)
            ),
            "upperArmAngleDriftAvg": (
                None if upperArmAngleDriftAvg is None else round(upperArmAngleDriftAvg, 2)
            ),
        }

        if upperArmAngleDriftAvg is None:
            warnings.append(
                "Upper-arm stability check was unavailable in many frames (hip landmark missing)."
            )

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Side tricep extension analysis completed.",
        )
