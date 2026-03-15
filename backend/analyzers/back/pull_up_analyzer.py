from collections import deque
from typing import Any, Deque, Dict, List, Optional

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class PullUpAnalyzer(BaseAnalyzer):
    """
    Back-view pull-up analyzer (full ROM rep count):
      - Counts a rep only after bottom -> top -> bottom
      - Bottom check uses near-lockout elbow angle
      - Top detection uses armpit angle if available, otherwise elbow angle fallback

    Current quality checks:
      - Did the rep start and finish from a deep enough bottom hang
      - Did the rep reach a high enough top position
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

        topElbowAngleDeg = float(options.get("topElbowAngleDeg", 95))
        topArmpitAngleDeg = float(options.get("topArmpitAngleDeg", 105))
        bottomElbowAngleDeg = float(options.get("bottomElbowAngleDeg", 170))
        leaveTopElbowDeg = float(options.get("leaveTopElbowDeg", 120))
        leaveBottomElbowDeg = float(options.get("leaveBottomElbowDeg", 155))
        hysteresisDeg = float(options.get("hysteresisDeg", 6))
        smoothWindow = int(options.get("smoothWindow", 5))
        requireArmpitForHeight = bool(options.get("requireArmpitForHeight", False))
        enableSymmetryCheck = bool(options.get("enableSymmetryCheck", True))
        topElbowDiffWarnDeg = float(options.get("topElbowDiffWarnDeg", 28))

        def avg2(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return (a + b) / 2.0

        repCount = 0
        issues = []
        repFeedback: List[Dict[str, Any]] = []
        warnings: List[str] = []

        elbowBuf: Deque[float] = deque(maxlen=max(1, smoothWindow))

        state = "UNKNOWN"  # BOTTOM_READY | UP_IN_REP | TOP_HIT | DOWN_TO_BOTTOM
        pendingRep: Optional[Dict[str, Any]] = None

        skippedMissingElbow = 0
        validElbowFrames = 0
        elbowMin = 999.0
        elbowMax = -999.0
        topHitFrames = 0
        bottomReadyFrames = 0
        armpitAvailableFrames = 0
        topElbowDiffSamples: List[float] = []

        for poseFrame in poseFrames:
            if not getattr(poseFrame, "hasPose", False):
                continue

            landmarks = poseFrame.landmarks

            leftElbow = getLandmarkAngle(landmarks, "left_shoulder", "left_elbow", "left_wrist")
            rightElbow = getLandmarkAngle(landmarks, "right_shoulder", "right_elbow", "right_wrist")
            elbowAngle = avg2(leftElbow, rightElbow)

            if elbowAngle is None:
                skippedMissingElbow += 1
                continue

            elbowBuf.append(float(elbowAngle))
            elbowSmoothed = sum(elbowBuf) / len(elbowBuf)

            validElbowFrames += 1
            elbowMin = min(elbowMin, elbowSmoothed)
            elbowMax = max(elbowMax, elbowSmoothed)

            leftArmpit = getLandmarkAngle(landmarks, "left_elbow", "left_shoulder", "left_hip")
            rightArmpit = getLandmarkAngle(landmarks, "right_elbow", "right_shoulder", "right_hip")
            armpitAngle = avg2(leftArmpit, rightArmpit)
            if armpitAngle is not None:
                armpitAvailableFrames += 1

            bottomHit = elbowSmoothed >= (bottomElbowAngleDeg - hysteresisDeg)
            leaveBottom = elbowSmoothed <= (leaveBottomElbowDeg + hysteresisDeg)
            elbowTopOk = elbowSmoothed <= (topElbowAngleDeg + hysteresisDeg)
            armpitTopOk = (
                armpitAngle is not None
                and float(armpitAngle) <= (topArmpitAngleDeg + hysteresisDeg)
            )
            topHit = armpitTopOk or elbowTopOk
            leaveTop = elbowSmoothed >= (leaveTopElbowDeg - hysteresisDeg)
            topElbowDiff = (
                abs(float(leftElbow) - float(rightElbow))
                if leftElbow is not None and rightElbow is not None
                else None
            )

            if bottomHit:
                bottomReadyFrames += 1
            if topHit:
                topHitFrames += 1

            if state == "UNKNOWN":
                state = "BOTTOM_READY" if bottomHit else "UP_IN_REP"

            if state == "BOTTOM_READY":
                if leaveBottom:
                    state = "UP_IN_REP"
                    pendingRep = {
                        "repIndex": repCount + 1,
                        "startFrameIndex": getattr(poseFrame, "frameIndex", None),
                        "heightOk": False,
                        "bottomOk": True,
                        "elbowDegAtTop": None,
                        "armpitDegAtTop": None,
                        "usedArmpitForTop": False,
                    }
                continue

            if state == "UP_IN_REP":
                if pendingRep is None:
                    pendingRep = {
                        "repIndex": repCount + 1,
                        "startFrameIndex": getattr(poseFrame, "frameIndex", None),
                        "heightOk": False,
                        "bottomOk": False,
                        "elbowDegAtTop": None,
                        "armpitDegAtTop": None,
                        "usedArmpitForTop": False,
                    }

                if topHit:
                    state = "TOP_HIT"
                    if requireArmpitForHeight:
                        heightOk = bool(armpitTopOk)
                    else:
                        heightOk = bool(armpitTopOk or elbowTopOk)
                    pendingRep["heightOk"] = bool(heightOk)
                    pendingRep["topFrameIndex"] = getattr(poseFrame, "frameIndex", None)
                    pendingRep["pauseFrameIndex"] = getattr(poseFrame, "frameIndex", None)
                    pendingRep["elbowDegAtTop"] = round(elbowSmoothed, 1)
                    pendingRep["armpitDegAtTop"] = (
                        None if armpitAngle is None else round(float(armpitAngle), 1)
                    )
                    pendingRep["usedArmpitForTop"] = bool(armpitTopOk)
                    pendingRep["topElbowDiffDeg"] = (
                        None if topElbowDiff is None else round(float(topElbowDiff), 1)
                    )
                elif bottomHit:
                    state = "BOTTOM_READY"
                    pendingRep = None
                continue

            if state == "TOP_HIT":
                if topHit and pendingRep is not None:
                    currentTopElbow = pendingRep.get("elbowDegAtTop")
                    if currentTopElbow is None or elbowSmoothed <= currentTopElbow:
                        pendingRep["topFrameIndex"] = getattr(poseFrame, "frameIndex", None)
                        pendingRep["pauseFrameIndex"] = getattr(poseFrame, "frameIndex", None)
                        pendingRep["elbowDegAtTop"] = round(elbowSmoothed, 1)
                        pendingRep["armpitDegAtTop"] = (
                            None if armpitAngle is None else round(float(armpitAngle), 1)
                        )
                        pendingRep["usedArmpitForTop"] = bool(armpitTopOk)
                        pendingRep["topElbowDiffDeg"] = (
                            None if topElbowDiff is None else round(float(topElbowDiff), 1)
                        )
                    continue

                if leaveTop:
                    state = "DOWN_TO_BOTTOM"
                continue

            if state == "DOWN_TO_BOTTOM":
                if bottomHit:
                    repCount += 1
                    state = "BOTTOM_READY"

                    repIssueCodes: List[str] = []
                    if pendingRep is not None and not pendingRep.get("bottomOk"):
                        repIssueCodes.append("bottom_incomplete")
                    if pendingRep is not None and not pendingRep.get("heightOk"):
                        repIssueCodes.append("height_incomplete")
                    symmetryOk = True
                    if enableSymmetryCheck and pendingRep is not None:
                        diff = pendingRep.get("topElbowDiffDeg")
                        symmetryOk = not isinstance(diff, (int, float)) or diff <= topElbowDiffWarnDeg
                        if not symmetryOk:
                            repIssueCodes.append("pull_asymmetry")

                    repQuality = 100.0
                    if pendingRep is not None and not pendingRep.get("bottomOk"):
                        repQuality -= 50.0
                    if pendingRep is not None and not pendingRep.get("heightOk"):
                        repQuality -= 50.0
                    if enableSymmetryCheck and not symmetryOk:
                        repQuality -= 15.0

                    if pendingRep is not None:
                        pendingRep["repIndex"] = repCount
                        pendingRep["endFrameIndex"] = getattr(poseFrame, "frameIndex", None)
                        pendingRep["quality"] = round(max(0.0, repQuality), 1)
                        pendingRep["issues"] = repIssueCodes
                        repFeedback.append(pendingRep)
                        diff = pendingRep.get("topElbowDiffDeg")
                        if isinstance(diff, (int, float)):
                            topElbowDiffSamples.append(float(diff))

                    pendingRep = None
                continue

        if repCount == 0:
            issues.append(
                self.buildIssue(
                    code="no_reps_detected",
                    message="No pull-up reps detected. Check pose quality or loosen the ROM thresholds.",
                    severity="medium",
                )
            )

        bottomIssueCount = sum(
            1 for rep in repFeedback if "bottom_incomplete" in (rep.get("issues") or [])
        )
        heightIssueCount = sum(
            1 for rep in repFeedback if "height_incomplete" in (rep.get("issues") or [])
        )
        symmetryIssueCount = sum(
            1 for rep in repFeedback if "pull_asymmetry" in (rep.get("issues") or [])
        )

        if repCount > 0 and bottomIssueCount > 0:
            issues.append(
                self.buildIssue(
                    code="bottom_position_issue",
                    message=f"Bottom hang depth looked incomplete in {bottomIssueCount}/{repCount} rep(s).",
                    severity="medium" if bottomIssueCount >= max(1, repCount // 2) else "low",
                )
            )

        if repCount > 0 and heightIssueCount > 0:
            issues.append(
                self.buildIssue(
                    code="top_position_issue",
                    message=f"Top position looked incomplete in {heightIssueCount}/{repCount} rep(s).",
                    severity="medium" if heightIssueCount >= max(1, repCount // 2) else "low",
                )
            )

        if repCount > 0 and symmetryIssueCount > 0:
            issues.append(
                self.buildIssue(
                    code="pull_asymmetry_issue",
                    message=f"Top pull symmetry looked off in {symmetryIssueCount}/{repCount} rep(s).",
                    severity="low",
                )
            )

        if pendingRep is not None:
            warnings.append(
                "The clip ended before one pull-up returned fully to the bottom position, so that final partial rep was not counted."
            )

        repScores: List[float] = [
            (1.0 if rep.get("bottomOk") else 0.0) * (1.0 if rep.get("heightOk") else 0.0)
            for rep in repFeedback
        ]
        summaryScore = (sum(repScores) / len(repScores)) if repScores else None

        metrics = {
            "topElbowAngleDeg": topElbowAngleDeg,
            "topArmpitAngleDeg": topArmpitAngleDeg,
            "bottomElbowAngleDeg": bottomElbowAngleDeg,
            "leaveTopElbowDeg": leaveTopElbowDeg,
            "leaveBottomElbowDeg": leaveBottomElbowDeg,
            "hysteresisDeg": hysteresisDeg,
            "smoothWindow": smoothWindow,
            "requireArmpitForHeight": requireArmpitForHeight,
            "enableSymmetryCheck": enableSymmetryCheck,
            "topElbowDiffWarnDeg": topElbowDiffWarnDeg,
            "skippedMissingElbowFrames": skippedMissingElbow,
            "validElbowFrames": validElbowFrames,
            "elbowAngleMin": None if validElbowFrames == 0 else round(elbowMin, 1),
            "elbowAngleMax": None if validElbowFrames == 0 else round(elbowMax, 1),
            "topHitFrames": topHitFrames,
            "bottomReadyFrames": bottomReadyFrames,
            "armpitAvailableFrames": armpitAvailableFrames,
            "topElbowDiffAvgDeg": (
                None
                if not topElbowDiffSamples
                else round(sum(topElbowDiffSamples) / len(topElbowDiffSamples), 1)
            ),
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Back pull-up analysis completed.",
        )
