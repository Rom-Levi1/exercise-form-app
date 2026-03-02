from typing import Any, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class FrontShoulderPressAnalyzer(BaseAnalyzer):
    """
    V1.0 front-view shoulder press analyzer:
    - Rep count (average left/right elbow angle state machine)
    - Bottom elbow bend check (did they lower enough?)
    - Top lockout / full press check (did they press high enough and straighten enough?)
    - Basic top-position symmetry check (are both sides level at the top?)

    Notes:
    - This is intentionally simple.
    - Front view is mainly used here for elbow angle, top reach, and symmetry.
    """

    def __init__(self):
        super().__init__("shoulder_press")

    def analyze(
        self,
        videoPath: str,
        poseFrames: List[Any],
        videoMetadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not poseFrames:
            return self.buildFailedResult(message="No pose frames provided.")

        options = options or {}

        # Rep detection thresholds
        topElbowThreshold = float(options.get("topElbowThreshold", 130.0))
        bottomElbowThreshold = float(options.get("bottomElbowThreshold", 95.0))

        # Per-rep quality thresholds
        bottomTooShallowThreshold = float(options.get("bottomTooShallowThreshold", 110.0))
        lockoutIncompleteThreshold = float(options.get("lockoutIncompleteThreshold", 145.0))
        minPressHeightThreshold = float(options.get("minPressHeightThreshold", 0.10))

        mildTopImbalanceThreshold = float(options.get("mildTopImbalanceThreshold", 0.08))
        highTopImbalanceThreshold = float(options.get("highTopImbalanceThreshold", 0.14))

        avgElbowAngles: List[Optional[float]] = []
        pressHeights: List[Optional[float]] = []
        topLevelDiffs: List[Optional[float]] = []

        for frame in poseFrames:
            if not frame.hasPose:
                avgElbowAngles.append(None)
                pressHeights.append(None)
                topLevelDiffs.append(None)
                continue

            landmarks = frame.landmarks

            avgElbowAngle = self._getAverageElbowAngle(landmarks)
            pressHeight = self._getPressHeightScore(landmarks)
            topLevelDiff = self._getTopLevelImbalance(landmarks)

            avgElbowAngles.append(avgElbowAngle)
            pressHeights.append(pressHeight)
            topLevelDiffs.append(topLevelDiff)

        validElbowAngles = [v for v in avgElbowAngles if v is not None]
        validPressHeights = [v for v in pressHeights if v is not None]
        validTopLevelDiffs = [v for v in topLevelDiffs if v is not None]

        totalFrames = len(poseFrames)
        validElbowFrames = len(validElbowAngles)
        validPressHeightFrames = len(validPressHeights)
        validTopLevelFrames = len(validTopLevelDiffs)

        if validElbowFrames < 20:
            return self.buildFailedResult(
                message="Not enough valid elbow-angle frames to analyze front-view shoulder press.",
                warnings=["Try a clearer front-view video with both arms fully visible."],
            )

        smoothedElbowAngles = self._movingAverageOptional(avgElbowAngles, windowSize=5)
        smoothedPressHeights = self._movingAverageOptional(pressHeights, windowSize=5)
        smoothedTopLevelDiffs = self._movingAverageOptional(topLevelDiffs, windowSize=5)

        repCount, repWindows = self._detectPressRepWindows(
            smoothedElbowAngles,
            topThreshold=topElbowThreshold,
            bottomThreshold=bottomElbowThreshold,
        )

        issues: List[Dict[str, Any]] = []
        warnings: List[str] = []
        repFeedback: List[Dict[str, Any]] = []

        validElbowRatio = validElbowFrames / totalFrames if totalFrames > 0 else 0.0
        validPressHeightRatio = validPressHeightFrames / totalFrames if totalFrames > 0 else 0.0
        validTopLevelRatio = validTopLevelFrames / totalFrames if totalFrames > 0 else 0.0

        if validElbowRatio < 0.7:
            warnings.append(
                "Arm tracking quality was low for many frames; shoulder press feedback may be less reliable."
            )

        if validPressHeightRatio < 0.6:
            warnings.append(
                "Wrist/shoulder tracking quality was low for many frames; top-reach feedback may be less reliable."
            )

        if validTopLevelRatio < 0.6:
            warnings.append(
                "Left/right wrist tracking quality was low for many frames; symmetry feedback may be less reliable."
            )

        perRepBottomElbowAngles: List[float] = []
        perRepTopElbowAngles: List[float] = []
        perRepTopPressHeights: List[float] = []
        perRepTopLevelDiffs: List[float] = []

        bottomBendIssueCount = 0
        lockoutIssueCount = 0
        topReachIssueCount = 0
        symmetryIssueCount = 0

        for repIndex, repWindow in enumerate(repWindows, start=1):
            repIssuesCodes: List[str] = []
            repQualityPenalties = 0.0

            startFrame = repWindow["startFrameIndex"]
            topFrame = repWindow["topFrameIndex"]
            endFrame = repWindow["endFrameIndex"]

            repElbowWindow = [
                value for value in smoothedElbowAngles[startFrame:endFrame + 1] if value is not None
            ]
            repPressHeightWindow = [
                value for value in smoothedPressHeights[startFrame:endFrame + 1] if value is not None
            ]
            topNeighborhoodStart = max(startFrame, topFrame - 2)
            topNeighborhoodEnd = min(endFrame, topFrame + 2)

            repTopLevelWindow = [
                value for value in smoothedTopLevelDiffs[topNeighborhoodStart:topNeighborhoodEnd + 1]
                if value is not None
            ]

            repBottomElbowAngle = min(repElbowWindow) if repElbowWindow else None
            repTopElbowAngle = max(repElbowWindow) if repElbowWindow else None
            repTopPressHeight = max(repPressHeightWindow) if repPressHeightWindow else None
            repTopLevelDiff = max(repTopLevelWindow) if repTopLevelWindow else None

            if repBottomElbowAngle is not None:
                perRepBottomElbowAngles.append(repBottomElbowAngle)

            if repTopElbowAngle is not None:
                perRepTopElbowAngles.append(repTopElbowAngle)

            if repTopPressHeight is not None:
                perRepTopPressHeights.append(repTopPressHeight)

            if repTopLevelDiff is not None:
                perRepTopLevelDiffs.append(repTopLevelDiff)

            # --- Bottom elbow bend ---
            # Lower elbow angle at the bottom means they lowered more.
            if repBottomElbowAngle is not None:
                if repBottomElbowAngle > bottomTooShallowThreshold:
                    repIssuesCodes.append("bottom_bend_shallow")
                    repQualityPenalties += 12
                    bottomBendIssueCount += 1

            # --- Top lockout / elbow straightening ---
            if repTopElbowAngle is not None:
                if repTopElbowAngle < lockoutIncompleteThreshold:
                    repIssuesCodes.append("lockout_incomplete")
                    repQualityPenalties += 12
                    lockoutIssueCount += 1

            # --- Did the weights move high enough? ---
            # Positive press height means wrists rose above shoulder level.
            if repTopPressHeight is not None:
                if repTopPressHeight < minPressHeightThreshold:
                    repIssuesCodes.append("top_reach_low")
                    repQualityPenalties += 12
                    topReachIssueCount += 1

            # --- Top-position symmetry ---
            if repTopLevelDiff is not None:
                if repTopLevelDiff >= highTopImbalanceThreshold:
                    repIssuesCodes.append("top_symmetry_high_imbalance")
                    repQualityPenalties += 15
                    symmetryIssueCount += 1
                elif repTopLevelDiff >= mildTopImbalanceThreshold:
                    repIssuesCodes.append("top_symmetry_mild_imbalance")
                    repQualityPenalties += 8
                    symmetryIssueCount += 1

            repQuality = max(0.0, min(100.0, 100.0 - repQualityPenalties))

            repFeedback.append(
                {
                    "rep": repIndex,
                    "quality": round(repQuality, 1),
                    "issues": repIssuesCodes,
                    "startFrameIndex": startFrame,
                    "topFrameIndex": topFrame,
                    "endFrameIndex": endFrame,
                    "checks": {
                        "elbowAngle": {
                            "bottomAvgElbowAngleDeg": round(repBottomElbowAngle, 2)
                            if repBottomElbowAngle is not None else None,
                            "topAvgElbowAngleDeg": round(repTopElbowAngle, 2)
                            if repTopElbowAngle is not None else None,
                        },
                        "topReach": {
                            "maxNormalizedWristAboveShoulder": round(repTopPressHeight, 4)
                            if repTopPressHeight is not None else None,
                        },
                        "symmetry": {
                            "topNormalizedWristLevelDiff": round(repTopLevelDiff, 4)
                            if repTopLevelDiff is not None else None,
                        },
                    },
                }
            )

        if repCount > 0:
            if bottomBendIssueCount > 0:
                severity = "medium" if bottomBendIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="bottom_bend_issue",
                        message=f"Elbow bend looked too shallow in {bottomBendIssueCount}/{repCount} rep(s).",
                        severity=severity,
                    )
                )

            if lockoutIssueCount > 0:
                severity = "medium" if lockoutIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="lockout_issue",
                        message=f"Top lockout looked incomplete in {lockoutIssueCount}/{repCount} rep(s).",
                        severity=severity,
                    )
                )

            if topReachIssueCount > 0:
                severity = "medium" if topReachIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="top_reach_issue",
                        message=f"The press did not move high enough in {topReachIssueCount}/{repCount} rep(s).",
                        severity=severity,
                    )
                )

            if symmetryIssueCount > 0:
                severity = "medium" if symmetryIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="asymmetry_issue",
                        message=f"Top position looked uneven in {symmetryIssueCount}/{repCount} rep(s).",
                        severity=severity,
                    )
                )

        if repCount == 0:
            warnings.append(
                "No shoulder press reps were confidently detected. Try a clearer front view and full arm visibility."
            )

        summaryScore = None
        repQualities = [rep["quality"] for rep in repFeedback if rep.get("quality") is not None]
        if repQualities:
            summaryScore = round(sum(repQualities) / len(repQualities), 1)

        minElbowAngle = min(validElbowAngles) if validElbowAngles else None
        maxElbowAngle = max(validElbowAngles) if validElbowAngles else None
        avgElbowAngle = (sum(validElbowAngles) / len(validElbowAngles)) if validElbowAngles else None

        minPressHeight = min(validPressHeights) if validPressHeights else None
        maxPressHeight = max(validPressHeights) if validPressHeights else None
        avgPressHeight = (sum(validPressHeights) / len(validPressHeights)) if validPressHeights else None

        minTopLevelDiff = min(validTopLevelDiffs) if validTopLevelDiffs else None
        maxTopLevelDiff = max(validTopLevelDiffs) if validTopLevelDiffs else None
        avgTopLevelDiff = (sum(validTopLevelDiffs) / len(validTopLevelDiffs)) if validTopLevelDiffs else None

        metrics = {
            "view": "front",
            "signalQuality": {
                "totalFrames": totalFrames,
                "validElbowFrames": validElbowFrames,
                "validElbowRatio": round(validElbowRatio, 4),
                "validPressHeightFrames": validPressHeightFrames,
                "validPressHeightRatio": round(validPressHeightRatio, 4),
                "validTopLevelFrames": validTopLevelFrames,
                "validTopLevelRatio": round(validTopLevelRatio, 4),
            },
            "globalSignals": {
                "avgLeftRightElbowAngleDeg": {
                    "min": round(minElbowAngle, 2) if minElbowAngle is not None else None,
                    "max": round(maxElbowAngle, 2) if maxElbowAngle is not None else None,
                    "avg": round(avgElbowAngle, 2) if avgElbowAngle is not None else None,
                },
                "normalizedWristAboveShoulder": {
                    "min": round(minPressHeight, 4) if minPressHeight is not None else None,
                    "max": round(maxPressHeight, 4) if maxPressHeight is not None else None,
                    "avg": round(avgPressHeight, 4) if avgPressHeight is not None else None,
                },
                "normalizedTopWristLevelDiff": {
                    "min": round(minTopLevelDiff, 4) if minTopLevelDiff is not None else None,
                    "max": round(maxTopLevelDiff, 4) if maxTopLevelDiff is not None else None,
                    "avg": round(avgTopLevelDiff, 4) if avgTopLevelDiff is not None else None,
                },
            },
            "repCheckSummary": {
                "bottomBendIssueCount": bottomBendIssueCount,
                "lockoutIssueCount": lockoutIssueCount,
                "topReachIssueCount": topReachIssueCount,
                "symmetryIssueCount": symmetryIssueCount,
                "perRepBottomElbowAngles": [round(v, 2) for v in perRepBottomElbowAngles],
                "perRepTopElbowAngles": [round(v, 2) for v in perRepTopElbowAngles],
                "perRepTopPressHeights": [round(v, 4) for v in perRepTopPressHeights],
                "perRepTopLevelDiffs": [round(v, 4) for v in perRepTopLevelDiffs],
            },
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Front-view shoulder press analysis completed (elbow angle + symmetry + top reach checks).",
        )

    def _getAverageElbowAngle(self, landmarks: dict) -> Optional[float]:
        """
        Average of left and right elbow angles:
        angle(shoulder, elbow, wrist)
        Falls back to whichever side is available.
        """
        leftElbowAngle = getLandmarkAngle(
            landmarks,
            "left_shoulder",
            "left_elbow",
            "left_wrist",
        )
        rightElbowAngle = getLandmarkAngle(
            landmarks,
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        )

        available = [v for v in [leftElbowAngle, rightElbowAngle] if v is not None]
        if not available:
            return None

        return sum(available) / len(available)

    def _getPressHeightScore(self, landmarks: dict) -> Optional[float]:
        """
        Returns how far the wrists rise above shoulder level, normalized by shoulder width.

        Positive values mean the wrists are above the shoulder line.
        Larger values mean a higher press.
        """
        leftWrist = landmarks.get("left_wrist")
        rightWrist = landmarks.get("right_wrist")
        leftShoulder = landmarks.get("left_shoulder")
        rightShoulder = landmarks.get("right_shoulder")

        if (
            leftWrist is None or rightWrist is None or
            leftShoulder is None or rightShoulder is None
        ):
            return None

        shoulderWidth = abs(rightShoulder.x - leftShoulder.x)
        if shoulderWidth <= 1e-9:
            return None

        shoulderCenterY = (leftShoulder.y + rightShoulder.y) / 2.0
        wristCenterY = (leftWrist.y + rightWrist.y) / 2.0

        # In image coordinates: smaller y = higher on screen
        return (shoulderCenterY - wristCenterY) / shoulderWidth

    def _getTopLevelImbalance(self, landmarks: dict) -> Optional[float]:
        """
        Returns left/right wrist height difference, normalized by shoulder width.
        Lower is better. Higher means the hands are uneven.
        """
        leftWrist = landmarks.get("left_wrist")
        rightWrist = landmarks.get("right_wrist")
        leftShoulder = landmarks.get("left_shoulder")
        rightShoulder = landmarks.get("right_shoulder")

        if (
            leftWrist is None or rightWrist is None or
            leftShoulder is None or rightShoulder is None
        ):
            return None

        shoulderWidth = abs(rightShoulder.x - leftShoulder.x)
        if shoulderWidth <= 1e-9:
            return None

        return abs(leftWrist.y - rightWrist.y) / shoulderWidth

    def _movingAverageOptional(
        self,
        values: List[Optional[float]],
        windowSize: int = 5,
    ) -> List[Optional[float]]:
        if windowSize <= 1:
            return values[:]

        half = windowSize // 2
        smoothed: List[Optional[float]] = []

        for i in range(len(values)):
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            window = [v for v in values[start:end] if v is not None]

            if not window:
                smoothed.append(None)
            else:
                smoothed.append(sum(window) / len(window))

        return smoothed

    def _detectPressRepWindows(
        self,
        elbowAngles: List[Optional[float]],
        topThreshold: float = 155.0,
        bottomThreshold: float = 95.0,
    ) -> Tuple[int, List[Dict[str, int]]]:
        """
        Detect shoulder press reps using average elbow angle.

        A rep is:
        bottom -> press up to top -> lower back to bottom

        Returns rep windows with:
        - startFrameIndex (bottom/start)
        - topFrameIndex
        - endFrameIndex (returned to bottom)
        """
        repWindows: List[Dict[str, int]] = []

        state = "unknown"
        repStartFrameIndex: Optional[int] = None
        topFrameIndex: Optional[int] = None
        lastBottomFrameIndex: Optional[int] = None

        for frameIndex, angle in enumerate(elbowAngles):
            if angle is None:
                continue

            if state == "unknown":
                if angle <= bottomThreshold:
                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                elif angle >= topThreshold:
                    state = "top"
                else:
                    state = "mid"

            elif state == "bottom":
                if angle > bottomThreshold + 5:
                    state = "up"
                    repStartFrameIndex = lastBottomFrameIndex if lastBottomFrameIndex is not None else frameIndex
                    topFrameIndex = None

            elif state == "up":
                if angle >= topThreshold:
                    state = "top"
                    topFrameIndex = frameIndex
                elif angle <= bottomThreshold:
                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                    repStartFrameIndex = None
                    topFrameIndex = None

            elif state == "top":
                if topFrameIndex is None:
                    topFrameIndex = frameIndex

                if angle < topThreshold - 5:
                    state = "down"

            elif state == "down":
                if angle >= topThreshold and topFrameIndex is None:
                    topFrameIndex = frameIndex

                if angle <= bottomThreshold:
                    if repStartFrameIndex is not None and topFrameIndex is not None:
                        repWindows.append(
                            {
                                "startFrameIndex": repStartFrameIndex,
                                "topFrameIndex": topFrameIndex,
                                "endFrameIndex": frameIndex,
                            }
                        )

                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                    repStartFrameIndex = None
                    topFrameIndex = None

                elif angle >= topThreshold:
                    # Went back to top without returning to bottom yet
                    state = "top"
                    if topFrameIndex is None:
                        topFrameIndex = frameIndex

            elif state == "mid":
                if angle <= bottomThreshold:
                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                elif angle >= topThreshold:
                    state = "top"

        return len(repWindows), repWindows