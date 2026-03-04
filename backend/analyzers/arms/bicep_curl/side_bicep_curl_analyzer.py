from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class SideBicepCurlAnalyzer(BaseAnalyzer):
    """
    Side view bicep curl analyzer:
      - Rep counting via elbow angle with hysteresis + smoothing
      - ROM validation per rep (min/max elbow angle + minimum ROM)
      - Elbow stability proxy (elbow position drift on x-axis relative to shoulder)
    """

    def __init__(self):
        super().__init__("bicep_curl_side")

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

        # Rep thresholds for curls (bottom=extended, top=flexed).
        BOTTOM_ANGLE = float(options.get("bottomAngleDeg", 158))
        TOP_ANGLE = float(options.get("topAngleDeg", 68))
        HYST_DEG = float(options.get("hysteresisDeg", 6))
        SMOOTH_WINDOW = int(options.get("smoothWindow", 5))

        # ROM thresholds
        MIN_ROM_DEG = float(options.get("minRomDeg", 70))
        BOTTOM_MARGIN = float(options.get("bottomMarginDeg", 12))
        TOP_MARGIN = float(options.get("topMarginDeg", 8))

        # Elbow steadiness threshold on x-axis (normalized image coordinates).
        ELBOW_REL_X_DRIFT_WARN = float(options.get("elbowRelXDriftWarn", 0.27))

        def get_xy(lm: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
            p = lm.get(name)
            if p is None:
                return None
            return (float(p.x), float(p.y))

        def safe_avg(values: List[float]) -> Optional[float]:
            return (sum(values) / len(values)) if values else None

        def in_bottom_zone(angle: float) -> bool:
            return angle >= (BOTTOM_ANGLE - HYST_DEG)

        def in_top_zone(angle: float) -> bool:
            return angle <= (TOP_ANGLE + HYST_DEG)

        repCount = 0
        issues: List[Dict[str, Any]] = []
        repFeedback: List[Dict[str, Any]] = []

        angleBuf: Deque[float] = deque(maxlen=max(1, SMOOTH_WINDOW))
        state = "UNKNOWN"  # "BOTTOM_READY" | "UP_IN_REP" | "TOP_LOCKED"

        rep_elbow_rel_x_drifts: List[float] = []

        rep_elbow_min: Optional[float] = None
        rep_elbow_max: Optional[float] = None
        rep_elbow_rel_x_min: Optional[float] = None
        rep_elbow_rel_x_max: Optional[float] = None
        rep_started_from_bottom = False

        for pf in poseFrames:
            if not getattr(pf, "hasPose", False):
                continue

            lm = pf.landmarks
            sh = get_xy(lm, SHOULDER)
            el = get_xy(lm, ELBOW)
            wr = get_xy(lm, WRIST)
            if not (sh and el and wr):
                continue

            rawElbowAngle = getLandmarkAngle(lm, SHOULDER, ELBOW, WRIST)
            if rawElbowAngle is None:
                continue

            angleBuf.append(float(rawElbowAngle))
            elbowAngle = sum(angleBuf) / len(angleBuf)
            elbowRelX = el[0] - sh[0]

            if state == "UNKNOWN":
                state = "BOTTOM_READY" if in_bottom_zone(elbowAngle) else "TOP_LOCKED"

            if state == "BOTTOM_READY":
                # Start a rep once the arm begins flexing away from the open/bottom zone.
                if not in_bottom_zone(elbowAngle):
                    state = "UP_IN_REP"
                    rep_elbow_min = elbowAngle
                    rep_elbow_max = elbowAngle
                    rep_elbow_rel_x_min = elbowRelX
                    rep_elbow_rel_x_max = elbowRelX
                    rep_started_from_bottom = True
                continue

            if state == "TOP_LOCKED":
                # Require returning to bottom before allowing next rep.
                if in_bottom_zone(elbowAngle):
                    state = "BOTTOM_READY"
                continue

            # state == UP_IN_REP
            rep_elbow_min = elbowAngle if rep_elbow_min is None else min(rep_elbow_min, elbowAngle)
            rep_elbow_max = elbowAngle if rep_elbow_max is None else max(rep_elbow_max, elbowAngle)
            rep_elbow_rel_x_min = (
                elbowRelX if rep_elbow_rel_x_min is None else min(rep_elbow_rel_x_min, elbowRelX)
            )
            rep_elbow_rel_x_max = (
                elbowRelX if rep_elbow_rel_x_max is None else max(rep_elbow_rel_x_max, elbowRelX)
            )

            # Count rep when reaching the top/flexed zone after starting from bottom.
            if in_top_zone(elbowAngle) and rep_started_from_bottom:
                repCount += 1

                rom = (
                    (rep_elbow_max - rep_elbow_min)
                    if (rep_elbow_min is not None and rep_elbow_max is not None)
                    else 0.0
                )
                hitTop = (rep_elbow_min is not None) and (rep_elbow_min <= (TOP_ANGLE + TOP_MARGIN))
                hitBottom = (rep_elbow_max is not None) and (
                    rep_elbow_max >= (BOTTOM_ANGLE - BOTTOM_MARGIN)
                )
                romOk = hitTop and hitBottom and (rom >= MIN_ROM_DEG)

                elbowRelXDrift = (
                    (rep_elbow_rel_x_max - rep_elbow_rel_x_min)
                    if (rep_elbow_rel_x_min is not None and rep_elbow_rel_x_max is not None)
                    else 0.0
                )
                elbowStable = elbowRelXDrift < ELBOW_REL_X_DRIFT_WARN
                rep_elbow_rel_x_drifts.append(elbowRelXDrift)

                repFeedback.append(
                    {
                        "repIndex": repCount,
                        "side": side,
                        "romOk": romOk,
                        "romDeg": round(rom, 1),
                        "minElbowDeg": None if rep_elbow_min is None else round(rep_elbow_min, 1),
                        "maxElbowDeg": None if rep_elbow_max is None else round(rep_elbow_max, 1),
                        "hitTop": hitTop,
                        "hitBottom": hitBottom,
                        "elbowStable": elbowStable,
                        "elbowRelXDrift": round(elbowRelXDrift, 3),
                    }
                )

                state = "TOP_LOCKED"
                rep_elbow_min = None
                rep_elbow_max = None
                rep_elbow_rel_x_min = None
                rep_elbow_rel_x_max = None
                rep_started_from_bottom = False

        elbowRelXDriftAvg = safe_avg(rep_elbow_rel_x_drifts)

        if repCount == 0:
            issues.append(
                self.buildIssue(
                    code="no_reps_detected",
                    message=(
                        "No side bicep curl reps detected. Ensure shoulder/elbow/wrist are visible "
                        "and tune thresholds if needed."
                    ),
                    severity="medium",
                )
            )

        if elbowRelXDriftAvg is not None and elbowRelXDriftAvg > ELBOW_REL_X_DRIFT_WARN:
            issues.append(
                self.buildIssue(
                    code="elbow_drift",
                    message=(
                        f"Elbow drift on x-axis looked high (avg drift={elbowRelXDriftAvg:.3f}). "
                        "Keep elbow fixed near torso."
                    ),
                    severity="low",
                )
            )

        def isGoodRep(rep: Dict[str, Any]) -> bool:
            return bool(rep.get("romOk") and rep.get("elbowStable"))

        goodReps = sum(1 for rep in repFeedback if isGoodRep(rep))
        summaryScore = (goodReps / repCount) if repCount > 0 else None

        metrics = {
            "side": side,
            "bottomAngleDeg": BOTTOM_ANGLE,
            "topAngleDeg": TOP_ANGLE,
            "hysteresisDeg": HYST_DEG,
            "smoothWindow": SMOOTH_WINDOW,
            "minRomDeg": MIN_ROM_DEG,
            "bottomMarginDeg": BOTTOM_MARGIN,
            "topMarginDeg": TOP_MARGIN,
            "elbowRelXDriftWarn": ELBOW_REL_X_DRIFT_WARN,
            "elbowRelXDriftAvg": (
                None if elbowRelXDriftAvg is None else round(elbowRelXDriftAvg, 4)
            ),
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=[],
            message="Side bicep curl analysis completed.",
        )
