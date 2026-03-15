from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class SideBicepCurlAnalyzer(BaseAnalyzer):
    """
    Side view bicep curl analyzer:
      - Rep counting via elbow angle with hysteresis + smoothing
      - More forgiving detection so messy curls still count as reps
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
        minDetectRomDeg = float(options.get("minDetectRomDeg", 35))
        detectionBottomSlackDeg = float(options.get("detectionBottomSlackDeg", 12))
        detectionTopSlackDeg = float(options.get("detectionTopSlackDeg", 12))
        ascentStartRomDeg = float(options.get("ascentStartRomDeg", 25))
        elbowRelXDriftWarn = float(options.get("elbowRelXDriftWarn", 0.27))
        upperArmAngleDriftWarn = float(options.get("upperArmAngleDriftWarn", 20))

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
                minDetectRomDeg=minDetectRomDeg,
                detectionBottomSlackDeg=detectionBottomSlackDeg,
                detectionTopSlackDeg=detectionTopSlackDeg,
                ascentStartRomDeg=ascentStartRomDeg,
                elbowRelXDriftWarn=elbowRelXDriftWarn,
                upperArmAngleDriftWarn=upperArmAngleDriftWarn,
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
        upperArmAngleDriftAvg = bestResult["upperArmAngleDriftAvg"]
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

        if upperArmAngleDriftAvg is not None and upperArmAngleDriftAvg > upperArmAngleDriftWarn:
            issues.append(
                self.buildIssue(
                    code="upper_arm_instability",
                    message=(
                        "Upper-arm angle changed a lot during curls "
                        f"(avg drift={upperArmAngleDriftAvg:.1f} deg)."
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
            "minDetectRomDeg": minDetectRomDeg,
            "detectionBottomSlackDeg": detectionBottomSlackDeg,
            "detectionTopSlackDeg": detectionTopSlackDeg,
            "ascentStartRomDeg": ascentStartRomDeg,
            "elbowRelXDriftWarn": elbowRelXDriftWarn,
            "upperArmAngleDriftWarn": upperArmAngleDriftWarn,
            "elbowRelXDriftAvg": (
                None if elbowRelXDriftAvg is None else round(elbowRelXDriftAvg, 4)
            ),
            "upperArmAngleDriftAvg": (
                None if upperArmAngleDriftAvg is None else round(upperArmAngleDriftAvg, 2)
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
        minDetectRomDeg: float,
        detectionBottomSlackDeg: float,
        detectionTopSlackDeg: float,
        ascentStartRomDeg: float,
        elbowRelXDriftWarn: float,
        upperArmAngleDriftWarn: float,
    ) -> Dict[str, Any]:
        shoulderName = f"{side}_shoulder"
        elbowName = f"{side}_elbow"
        wristName = f"{side}_wrist"
        hipName = f"{side}_hip"

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

        def in_detection_bottom_zone(angle: float) -> bool:
            return angle >= (bottomAngleDeg - hysteresisDeg - detectionBottomSlackDeg)

        def in_detection_top_zone(angle: float) -> bool:
            return angle <= (topAngleDeg + hysteresisDeg + detectionTopSlackDeg)

        repCount = 0
        repFeedback: List[Dict[str, Any]] = []

        angleBuf: Deque[float] = deque(maxlen=max(1, smoothWindow))
        state = "UNKNOWN"  # BOTTOM_READY | UP_IN_REP | TOP_FINISH | DOWN_TO_BOTTOM

        repElbowRelXDrifts: List[float] = []
        repUpperArmAngleDrifts: List[float] = []
        trackedFrameCount = 0
        repRomValues: List[float] = []

        repElbowMin: Optional[float] = None
        repElbowMax: Optional[float] = None
        repElbowRelXMin: Optional[float] = None
        repElbowRelXMax: Optional[float] = None
        repStartFrameIndex: Optional[int] = None
        repTopFrameIndex: Optional[int] = None
        repEndFrameIndex: Optional[int] = None
        bottomReadyPeakAngle: Optional[float] = None
        repUpperArmAngleMin: Optional[float] = None
        repUpperArmAngleMax: Optional[float] = None

        def finalize_rep() -> None:
            nonlocal repCount
            nonlocal repElbowMin
            nonlocal repElbowMax
            nonlocal repElbowRelXMin
            nonlocal repElbowRelXMax
            nonlocal repStartFrameIndex
            nonlocal repTopFrameIndex
            nonlocal repEndFrameIndex
            nonlocal repUpperArmAngleMin
            nonlocal repUpperArmAngleMax

            if repStartFrameIndex is None:
                return

            rom = (
                (repElbowMax - repElbowMin)
                if (repElbowMin is not None and repElbowMax is not None)
                else 0.0
            )
            if rom < minDetectRomDeg:
                repElbowMin = None
                repElbowMax = None
                repElbowRelXMin = None
                repElbowRelXMax = None
                repStartFrameIndex = None
                repTopFrameIndex = None
                repEndFrameIndex = None
                repUpperArmAngleMin = None
                repUpperArmAngleMax = None
                return

            repCount += 1
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

            upperArmAngleDrift = (
                (repUpperArmAngleMax - repUpperArmAngleMin)
                if (repUpperArmAngleMin is not None and repUpperArmAngleMax is not None)
                else None
            )
            upperArmStable = (
                None if upperArmAngleDrift is None else (upperArmAngleDrift < upperArmAngleDriftWarn)
            )
            if upperArmAngleDrift is not None:
                repUpperArmAngleDrifts.append(upperArmAngleDrift)

            repIssuesCodes: List[str] = []
            repQualityPenalty = 0.0
            if not hitBottom:
                repIssuesCodes.append("bottom_position_incomplete")
                repQualityPenalty += 20.0
            if not hitTop:
                repIssuesCodes.append("top_position_incomplete")
                repQualityPenalty += 20.0
            if hitBottom and hitTop and rom < minRomDeg:
                repIssuesCodes.append("rom_incomplete")
                repQualityPenalty += 10.0
            if not elbowStable:
                repIssuesCodes.append("elbow_drift")
                repQualityPenalty += 30.0
            if upperArmStable is False:
                repIssuesCodes.append("upper_arm_instability")
                repQualityPenalty += 20.0

            repQuality = max(0.0, 100.0 - repQualityPenalty)

            repFeedback.append(
                {
                    "repIndex": repCount,
                    "side": side,
                    "startFrameIndex": repStartFrameIndex,
                    "pauseFrameIndex": repTopFrameIndex,
                    "endFrameIndex": repEndFrameIndex,
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
                    "upperArmStable": upperArmStable,
                    "upperArmAngleDriftDeg": (
                        None if upperArmAngleDrift is None else round(upperArmAngleDrift, 1)
                    ),
                }
            )

            repElbowMin = None
            repElbowMax = None
            repElbowRelXMin = None
            repElbowRelXMax = None
            repStartFrameIndex = None
            repTopFrameIndex = None
            repEndFrameIndex = None
            repUpperArmAngleMin = None
            repUpperArmAngleMax = None

        for poseFrame in poseFrames:
            if not getattr(poseFrame, "hasPose", False):
                continue

            landmarks = poseFrame.landmarks
            shoulder = get_xy(landmarks, shoulderName)
            elbow = get_xy(landmarks, elbowName)
            wrist = get_xy(landmarks, wristName)
            hip = get_xy(landmarks, hipName)
            if not (shoulder and elbow and wrist):
                continue

            rawElbowAngle = getLandmarkAngle(landmarks, shoulderName, elbowName, wristName)
            if rawElbowAngle is None:
                continue

            trackedFrameCount += 1
            angleBuf.append(float(rawElbowAngle))
            elbowAngle = sum(angleBuf) / len(angleBuf)
            elbowRelX = elbow[0] - shoulder[0]
            upperArmAngle = (
                getLandmarkAngle(landmarks, elbowName, shoulderName, hipName)
                if hip is not None
                else None
            )

            if state == "UNKNOWN":
                state = "BOTTOM_READY" if in_detection_bottom_zone(elbowAngle) else "UP_IN_REP"
                if state == "UP_IN_REP":
                    repElbowMin = elbowAngle
                    repElbowMax = elbowAngle
                    repElbowRelXMin = elbowRelX
                    repElbowRelXMax = elbowRelX
                    repStartFrameIndex = getattr(poseFrame, "frameIndex", None)
                    repTopFrameIndex = getattr(poseFrame, "frameIndex", None)
                    repEndFrameIndex = getattr(poseFrame, "frameIndex", None)
                    repUpperArmAngleMin = float(upperArmAngle) if upperArmAngle is not None else None
                    repUpperArmAngleMax = float(upperArmAngle) if upperArmAngle is not None else None
                else:
                    bottomReadyPeakAngle = elbowAngle

            if state == "BOTTOM_READY":
                bottomReadyPeakAngle = (
                    elbowAngle
                    if bottomReadyPeakAngle is None
                    else max(bottomReadyPeakAngle, elbowAngle)
                )
                if (
                    in_detection_top_zone(elbowAngle)
                    or (
                        bottomReadyPeakAngle is not None
                        and (bottomReadyPeakAngle - elbowAngle) >= ascentStartRomDeg
                    )
                ):
                    state = "UP_IN_REP"
                    repElbowMin = elbowAngle
                    repElbowMax = elbowAngle
                    repElbowRelXMin = elbowRelX
                    repElbowRelXMax = elbowRelX
                    repStartFrameIndex = getattr(poseFrame, "frameIndex", None)
                    repTopFrameIndex = getattr(poseFrame, "frameIndex", None)
                    repEndFrameIndex = getattr(poseFrame, "frameIndex", None)
                    repUpperArmAngleMin = float(upperArmAngle) if upperArmAngle is not None else None
                    repUpperArmAngleMax = float(upperArmAngle) if upperArmAngle is not None else None
                    bottomReadyPeakAngle = None
                continue

            if state == "TOP_FINISH":
                if repElbowMin is None or elbowAngle <= repElbowMin:
                    repElbowMin = elbowAngle
                    repTopFrameIndex = getattr(poseFrame, "frameIndex", None)
                repEndFrameIndex = getattr(poseFrame, "frameIndex", None)

                if in_detection_top_zone(elbowAngle):
                    continue

                state = "DOWN_TO_BOTTOM"
                continue

            if state == "DOWN_TO_BOTTOM":
                repElbowMin = elbowAngle if repElbowMin is None else min(repElbowMin, elbowAngle)
                repElbowMax = elbowAngle if repElbowMax is None else max(repElbowMax, elbowAngle)
                repElbowRelXMin = (
                    elbowRelX if repElbowRelXMin is None else min(repElbowRelXMin, elbowRelX)
                )
                repElbowRelXMax = (
                    elbowRelX if repElbowRelXMax is None else max(repElbowRelXMax, elbowRelX)
                )
                repEndFrameIndex = getattr(poseFrame, "frameIndex", None)
                if upperArmAngle is not None:
                    repUpperArmAngleMin = (
                        float(upperArmAngle)
                        if repUpperArmAngleMin is None
                        else min(repUpperArmAngleMin, float(upperArmAngle))
                    )
                    repUpperArmAngleMax = (
                        float(upperArmAngle)
                        if repUpperArmAngleMax is None
                        else max(repUpperArmAngleMax, float(upperArmAngle))
                    )

                if in_detection_bottom_zone(elbowAngle):
                    finalize_rep()
                    state = "BOTTOM_READY"
                    bottomReadyPeakAngle = elbowAngle
                elif in_detection_top_zone(elbowAngle):
                    state = "TOP_FINISH"
                continue

            repElbowMin = elbowAngle if repElbowMin is None else min(repElbowMin, elbowAngle)
            repElbowMax = elbowAngle if repElbowMax is None else max(repElbowMax, elbowAngle)
            repElbowRelXMin = elbowRelX if repElbowRelXMin is None else min(repElbowRelXMin, elbowRelX)
            repElbowRelXMax = elbowRelX if repElbowRelXMax is None else max(repElbowRelXMax, elbowRelX)
            repEndFrameIndex = getattr(poseFrame, "frameIndex", None)
            if upperArmAngle is not None:
                repUpperArmAngleMin = (
                    float(upperArmAngle)
                    if repUpperArmAngleMin is None
                    else min(repUpperArmAngleMin, float(upperArmAngle))
                )
                repUpperArmAngleMax = (
                    float(upperArmAngle)
                    if repUpperArmAngleMax is None
                    else max(repUpperArmAngleMax, float(upperArmAngle))
                )

            if in_detection_top_zone(elbowAngle) and (
                repElbowMax is not None
                and repElbowMin is not None
                and (repElbowMax - repElbowMin) >= minDetectRomDeg
            ):
                state = "TOP_FINISH"
                repTopFrameIndex = getattr(poseFrame, "frameIndex", None)
                continue

        if state in {"TOP_FINISH", "DOWN_TO_BOTTOM"}:
            finalize_rep()

        goodReps = sum(
            1 for rep in repFeedback if rep.get("romOk") and rep.get("elbowStable")
        )
        elbowRelXDriftAvg = safe_avg(repElbowRelXDrifts)

        return {
            "side": side,
            "repCount": repCount,
            "goodReps": goodReps,
            "repFeedback": repFeedback,
            "elbowRelXDriftAvg": elbowRelXDriftAvg,
            "upperArmAngleDriftAvg": safe_avg(repUpperArmAngleDrifts),
            "trackedFrameCount": trackedFrameCount,
            "avgRomDeg": safe_avg(repRomValues) or 0.0,
        }
