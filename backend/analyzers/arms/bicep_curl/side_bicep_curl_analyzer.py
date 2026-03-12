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

    Supports automatic side selection so left-arm side videos work without an explicit side choice.
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

        bottomAngleDeg = float(options.get("bottomAngleDeg", 158))
        topAngleDeg = float(options.get("topAngleDeg", 68))
        hysteresisDeg = float(options.get("hysteresisDeg", 6))
        smoothWindow = int(options.get("smoothWindow", 5))
        minRomDeg = float(options.get("minRomDeg", 70))
        bottomMarginDeg = float(options.get("bottomMarginDeg", 12))
        topMarginDeg = float(options.get("topMarginDeg", 8))
        elbowRelXDriftWarn = float(options.get("elbowRelXDriftWarn", 0.27))

        requestedSide = str(options.get("side", "auto")).lower()
        if requestedSide not in ("left", "right", "auto"):
            requestedSide = "auto"

        candidateSides = ("left", "right") if requestedSide == "auto" else (requestedSide,)
        candidateResults = [
            self._analyze_side(
                poseFrames=poseFrames,
                side=side,
                bottomAngleDeg=bottomAngleDeg,
                topAngleDeg=topAngleDeg,
                hysteresisDeg=hysteresisDeg,
                smoothWindow=smoothWindow,
                minRomDeg=minRomDeg,
                bottomMarginDeg=bottomMarginDeg,
                topMarginDeg=topMarginDeg,
                elbowRelXDriftWarn=elbowRelXDriftWarn,
            )
            for side in candidateSides
        ]

        bestResult = max(
            candidateResults,
            key=lambda result: (
                result["repCount"],
                result["goodReps"],
                result["trackedFrameCount"],
                result["avgRomDeg"],
            ),
        )

        repCount = bestResult["repCount"]
        repFeedback = bestResult["repFeedback"]
        elbowRelXDriftAvg = bestResult["elbowRelXDriftAvg"]
        chosenSide = bestResult["side"]

        issues: List[Dict[str, Any]] = []
        warnings: List[str] = []

        if repCount == 0:
            issues.append(
                self.buildIssue(
                    code="no_reps_detected",
                    message=(
                        "No side bicep curl reps detected. Ensure the working shoulder, elbow, and wrist "
                        "are clearly visible through the full curl."
                    ),
                    severity="medium",
                )
            )

        if elbowRelXDriftAvg is not None and elbowRelXDriftAvg > elbowRelXDriftWarn:
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

        if requestedSide == "auto":
            if repCount > 0:
                warnings.append(
                    f"Automatically analyzed the {chosenSide} arm based on clearer curl tracking."
                )
            else:
                warnings.append(
                    "Tried analyzing both arms automatically, but no clear reps were found."
                )

        goodReps = bestResult["goodReps"]
        summaryScore = (goodReps / repCount) if repCount > 0 else None

        metrics = {
            "side": chosenSide,
            "sideSelectionMode": requestedSide,
            "bottomAngleDeg": bottomAngleDeg,
            "topAngleDeg": topAngleDeg,
            "hysteresisDeg": hysteresisDeg,
            "smoothWindow": smoothWindow,
            "minRomDeg": minRomDeg,
            "bottomMarginDeg": bottomMarginDeg,
            "topMarginDeg": topMarginDeg,
            "elbowRelXDriftWarn": elbowRelXDriftWarn,
            "elbowRelXDriftAvg": (
                None if elbowRelXDriftAvg is None else round(elbowRelXDriftAvg, 4)
            ),
            "trackedFrameCount": bestResult["trackedFrameCount"],
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Side bicep curl analysis completed.",
        )

    def _analyze_side(
        self,
        poseFrames: List[Any],
        side: str,
        bottomAngleDeg: float,
        topAngleDeg: float,
        hysteresisDeg: float,
        smoothWindow: int,
        minRomDeg: float,
        bottomMarginDeg: float,
        topMarginDeg: float,
        elbowRelXDriftWarn: float,
    ) -> Dict[str, Any]:
        shoulderName = f"{side}_shoulder"
        elbowName = f"{side}_elbow"
        wristName = f"{side}_wrist"

        def get_xy(lm: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
            point = lm.get(name)
            if point is None:
                return None
            return (float(point.x), float(point.y))

        def safe_avg(values: List[float]) -> Optional[float]:
            return (sum(values) / len(values)) if values else None

        def in_bottom_zone(angle: float) -> bool:
            return angle >= (bottomAngleDeg - hysteresisDeg)

        def in_top_zone(angle: float) -> bool:
            return angle <= (topAngleDeg + hysteresisDeg)

        repCount = 0
        repFeedback: List[Dict[str, Any]] = []

        angleBuf: Deque[float] = deque(maxlen=max(1, smoothWindow))
        state = "UNKNOWN"

        repElbowRelXDrifts: List[float] = []
        trackedFrameCount = 0
        repRomValues: List[float] = []

        repElbowMin: Optional[float] = None
        repElbowMax: Optional[float] = None
        repElbowRelXMin: Optional[float] = None
        repElbowRelXMax: Optional[float] = None
        repStartedFromBottom = False
        repStartFrameIndex: Optional[int] = None

        for poseFrame in poseFrames:
            if not getattr(poseFrame, "hasPose", False):
                continue

            landmarks = poseFrame.landmarks
            shoulder = get_xy(landmarks, shoulderName)
            elbow = get_xy(landmarks, elbowName)
            wrist = get_xy(landmarks, wristName)
            if not (shoulder and elbow and wrist):
                continue

            rawElbowAngle = getLandmarkAngle(landmarks, shoulderName, elbowName, wristName)
            if rawElbowAngle is None:
                continue

            trackedFrameCount += 1
            angleBuf.append(float(rawElbowAngle))
            elbowAngle = sum(angleBuf) / len(angleBuf)
            elbowRelX = elbow[0] - shoulder[0]

            if state == "UNKNOWN":
                state = "BOTTOM_READY" if in_bottom_zone(elbowAngle) else "TOP_LOCKED"

            if state == "BOTTOM_READY":
                if not in_bottom_zone(elbowAngle):
                    state = "UP_IN_REP"
                    repElbowMin = elbowAngle
                    repElbowMax = elbowAngle
                    repElbowRelXMin = elbowRelX
                    repElbowRelXMax = elbowRelX
                    repStartedFromBottom = True
                    repStartFrameIndex = getattr(poseFrame, "frameIndex", None)
                continue

            if state == "TOP_LOCKED":
                if in_bottom_zone(elbowAngle):
                    state = "BOTTOM_READY"
                continue

            repElbowMin = elbowAngle if repElbowMin is None else min(repElbowMin, elbowAngle)
            repElbowMax = elbowAngle if repElbowMax is None else max(repElbowMax, elbowAngle)
            repElbowRelXMin = elbowRelX if repElbowRelXMin is None else min(repElbowRelXMin, elbowRelX)
            repElbowRelXMax = elbowRelX if repElbowRelXMax is None else max(repElbowRelXMax, elbowRelX)

            if in_top_zone(elbowAngle) and repStartedFromBottom:
                repCount += 1

                rom = (
                    (repElbowMax - repElbowMin)
                    if (repElbowMin is not None and repElbowMax is not None)
                    else 0.0
                )
                repRomValues.append(rom)
                hitTop = (repElbowMin is not None) and (repElbowMin <= (topAngleDeg + topMarginDeg))
                hitBottom = (repElbowMax is not None) and (repElbowMax >= (bottomAngleDeg - bottomMarginDeg))
                romOk = hitTop and hitBottom and (rom >= minRomDeg)

                elbowRelXDrift = (
                    (repElbowRelXMax - repElbowRelXMin)
                    if (repElbowRelXMin is not None and repElbowRelXMax is not None)
                    else 0.0
                )
                elbowStable = elbowRelXDrift < elbowRelXDriftWarn
                repElbowRelXDrifts.append(elbowRelXDrift)

                repIssuesCodes: List[str] = []
                repQualityPenalty = 0.0
                if not romOk:
                    repIssuesCodes.append("rom_incomplete")
                    repQualityPenalty += 50.0
                if not elbowStable:
                    repIssuesCodes.append("elbow_drift")
                    repQualityPenalty += 50.0

                repQuality = max(0.0, 100.0 - repQualityPenalty)

                repFeedback.append(
                    {
                        "repIndex": repCount,
                        "side": side,
                        "startFrameIndex": repStartFrameIndex,
                        "endFrameIndex": getattr(poseFrame, "frameIndex", None),
                        "quality": round(repQuality, 1),
                        "issues": repIssuesCodes,
                        "romOk": romOk,
                        "romDeg": round(rom, 1),
                        "minElbowDeg": None if repElbowMin is None else round(repElbowMin, 1),
                        "maxElbowDeg": None if repElbowMax is None else round(repElbowMax, 1),
                        "hitTop": hitTop,
                        "hitBottom": hitBottom,
                        "elbowStable": elbowStable,
                        "elbowRelXDrift": round(elbowRelXDrift, 3),
                    }
                )

                state = "TOP_LOCKED"
                repElbowMin = None
                repElbowMax = None
                repElbowRelXMin = None
                repElbowRelXMax = None
                repStartedFromBottom = False
                repStartFrameIndex = None

        goodReps = sum(1 for rep in repFeedback if rep.get("romOk") and rep.get("elbowStable"))
        elbowRelXDriftAvg = safe_avg(repElbowRelXDrifts)

        return {
            "side": side,
            "repCount": repCount,
            "goodReps": goodReps,
            "repFeedback": repFeedback,
            "elbowRelXDriftAvg": elbowRelXDriftAvg,
            "trackedFrameCount": trackedFrameCount,
            "avgRomDeg": safe_avg(repRomValues) or 0.0,
        }
