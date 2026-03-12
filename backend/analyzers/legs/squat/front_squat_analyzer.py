from typing import Any, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer


class FrontSquatAnalyzer(BaseAnalyzer):
    """
    Front-view squat analyzer:
    - Rep counting from vertical hip-center motion
    - Stance-width check per rep
    - Left/right symmetry check per rep

    Front view remains a supplemental squat angle, but now returns real rep windows
    so the UI and feedback video are consistent with the rest of the product.
    """

    def __init__(self):
        super().__init__("squat")

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

        stanceTooNarrowRatio = float(options.get("stanceTooNarrowRatio", 0.9))
        stanceTooWideRatio = float(options.get("stanceTooWideRatio", 1.8))
        mildAsymmetryThreshold = float(options.get("mildAsymmetryThreshold", 0.08))
        highAsymmetryThreshold = float(options.get("highAsymmetryThreshold", 0.14))

        hipCenterYs: List[Optional[float]] = []
        stanceRatios: List[Optional[float]] = []
        symmetryScores: List[Optional[float]] = []

        for frame in poseFrames:
            if not frame.hasPose:
                hipCenterYs.append(None)
                stanceRatios.append(None)
                symmetryScores.append(None)
                continue

            landmarks = frame.landmarks
            hipCenterYs.append(self._getHipCenterY(landmarks))
            stanceRatios.append(self._getStanceWidthRatio(landmarks))
            symmetryScores.append(self._getFrontSymmetryScore(landmarks))

        validHipCenterYs = [value for value in hipCenterYs if value is not None]
        validStanceRatios = [value for value in stanceRatios if value is not None]
        validSymmetryScores = [value for value in symmetryScores if value is not None]

        totalFrames = len(poseFrames)
        validHipFrames = len(validHipCenterYs)
        validStanceFrames = len(validStanceRatios)
        validSymmetryFrames = len(validSymmetryScores)

        if validHipFrames < 20:
            return self.buildFailedResult(
                message="Not enough valid front-view frames to detect squat reps.",
                warnings=["Try a clearer front-view video with the hips and ankles fully visible."],
            )

        smoothedHipCenterYs = self._movingAverageOptional(hipCenterYs, windowSize=5)
        smoothedStanceRatios = self._movingAverageOptional(stanceRatios, windowSize=5)
        smoothedSymmetryScores = self._movingAverageOptional(symmetryScores, windowSize=5)

        repCount, repWindows, motionRange = self._detectRepWindows(smoothedHipCenterYs)

        issues: List[Dict[str, Any]] = []
        warnings: List[str] = []
        repFeedback: List[Dict[str, Any]] = []
        visualEvents: List[Dict[str, Any]] = []

        validHipRatio = validHipFrames / totalFrames if totalFrames > 0 else 0.0
        validStanceRatio = validStanceFrames / totalFrames if totalFrames > 0 else 0.0
        validSymmetryRatio = validSymmetryFrames / totalFrames if totalFrames > 0 else 0.0

        if validHipRatio < 0.7:
            warnings.append(
                "Vertical body tracking was unstable in many frames; front squat rep counting may be less reliable."
            )
        if validStanceRatio < 0.6:
            warnings.append(
                "Foot/shoulder tracking quality was low for many frames; stance-width feedback may be less reliable."
            )
        if validSymmetryRatio < 0.6:
            warnings.append(
                "Left/right body tracking quality was low for many frames; symmetry feedback may be less reliable."
            )

        if motionRange < 0.035:
            warnings.append(
                "Vertical squat motion looked limited in this clip, so rep detection may be conservative."
            )

        stanceIssueCount = 0
        symmetryIssueCount = 0
        perRepStanceRatios: List[float] = []
        perRepSymmetryScores: List[float] = []

        for repIndex, repWindow in enumerate(repWindows, start=1):
            startFrame = repWindow["startFrameIndex"]
            bottomFrame = repWindow["bottomFrameIndex"]
            endFrame = repWindow["endFrameIndex"]

            repStanceWindow = [
                value for value in smoothedStanceRatios[startFrame:endFrame + 1] if value is not None
            ]
            repSymmetryWindow = [
                value for value in smoothedSymmetryScores[startFrame:endFrame + 1] if value is not None
            ]

            repAvgStanceRatio = (
                sum(repStanceWindow) / len(repStanceWindow) if repStanceWindow else None
            )
            repMaxSymmetryScore = max(repSymmetryWindow) if repSymmetryWindow else None

            if repAvgStanceRatio is not None:
                perRepStanceRatios.append(repAvgStanceRatio)
            if repMaxSymmetryScore is not None:
                perRepSymmetryScores.append(repMaxSymmetryScore)

            repIssueCodes: List[str] = []
            repQualityPenalty = 0.0

            stanceClassification = "unknown"
            if repAvgStanceRatio is not None:
                if repAvgStanceRatio < stanceTooNarrowRatio:
                    stanceClassification = "too_narrow"
                    repIssueCodes.append("stance_too_narrow")
                    repQualityPenalty += 12.0
                    stanceIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "stance_too_narrow",
                            "message": "Stance looks too narrow in this rep.",
                            "severity": "medium",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                        }
                    )
                elif repAvgStanceRatio > stanceTooWideRatio:
                    stanceClassification = "too_wide"
                    repIssueCodes.append("stance_too_wide")
                    repQualityPenalty += 10.0
                    stanceIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "stance_too_wide",
                            "message": "Stance looks too wide in this rep.",
                            "severity": "low",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                        }
                    )
                else:
                    stanceClassification = "acceptable"

            symmetryClassification = "unknown"
            if repMaxSymmetryScore is not None:
                if repMaxSymmetryScore >= highAsymmetryThreshold:
                    symmetryClassification = "high_imbalance"
                    repIssueCodes.append("symmetry_high_imbalance")
                    repQualityPenalty += 15.0
                    symmetryIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "symmetry_high_imbalance",
                            "message": "Clear left-right imbalance in this rep.",
                            "severity": "medium",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                        }
                    )
                elif repMaxSymmetryScore >= mildAsymmetryThreshold:
                    symmetryClassification = "mild_imbalance"
                    repIssueCodes.append("symmetry_mild_imbalance")
                    repQualityPenalty += 8.0
                    symmetryIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "symmetry_mild_imbalance",
                            "message": "Mild left-right imbalance in this rep.",
                            "severity": "low",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                        }
                    )
                else:
                    symmetryClassification = "balanced"

            repQuality = max(0.0, min(100.0, 100.0 - repQualityPenalty))
            repFeedback.append(
                {
                    "rep": repIndex,
                    "quality": round(repQuality, 1),
                    "issues": repIssueCodes,
                    "startFrameIndex": startFrame,
                    "bottomFrameIndex": bottomFrame,
                    "endFrameIndex": endFrame,
                    "checks": {
                        "stanceWidth": {
                            "classification": stanceClassification,
                            "avgStanceToShoulderRatio": round(repAvgStanceRatio, 3)
                            if repAvgStanceRatio is not None else None,
                        },
                        "symmetry": {
                            "classification": symmetryClassification,
                            "maxNormalizedImbalance": round(repMaxSymmetryScore, 4)
                            if repMaxSymmetryScore is not None else None,
                        },
                    },
                }
            )

        if repCount == 0:
            warnings.append(
                "No front-view squat reps were confidently detected. Try a clearer clip with a full top-bottom-top motion."
            )
        else:
            if stanceIssueCount > 0:
                issues.append(
                    self.buildIssue(
                        code="stance_width_issue",
                        message=f"Stance width looked off in {stanceIssueCount}/{repCount} rep(s).",
                        severity="medium" if stanceIssueCount >= max(1, repCount // 2) else "low",
                    )
                )
            if symmetryIssueCount > 0:
                issues.append(
                    self.buildIssue(
                        code="asymmetry_issue",
                        message=f"Left/right imbalance appeared in {symmetryIssueCount}/{repCount} rep(s).",
                        severity="medium" if symmetryIssueCount >= max(1, repCount // 2) else "low",
                    )
                )

        repQualities = [rep["quality"] for rep in repFeedback if rep.get("quality") is not None]
        summaryScore = round(sum(repQualities) / len(repQualities), 1) if repQualities else None

        minStanceRatio = min(validStanceRatios) if validStanceRatios else None
        maxStanceRatio = max(validStanceRatios) if validStanceRatios else None
        avgStanceRatio = (
            sum(validStanceRatios) / len(validStanceRatios) if validStanceRatios else None
        )
        minSymmetryScore = min(validSymmetryScores) if validSymmetryScores else None
        maxSymmetryScore = max(validSymmetryScores) if validSymmetryScores else None
        avgSymmetryScore = (
            sum(validSymmetryScores) / len(validSymmetryScores) if validSymmetryScores else None
        )

        metrics = {
            "view": "front",
            "analysisMode": "rep_level",
            "visualFeedback": {
                "events": visualEvents,
            },
            "signalQuality": {
                "totalFrames": totalFrames,
                "validHipFrames": validHipFrames,
                "validHipRatio": round(validHipRatio, 4),
                "validStanceFrames": validStanceFrames,
                "validStanceRatio": round(validStanceRatio, 4),
                "validSymmetryFrames": validSymmetryFrames,
                "validSymmetryRatio": round(validSymmetryRatio, 4),
            },
            "globalSignals": {
                "hipCenterY": {
                    "min": round(min(validHipCenterYs), 4) if validHipCenterYs else None,
                    "max": round(max(validHipCenterYs), 4) if validHipCenterYs else None,
                    "avg": round(sum(validHipCenterYs) / len(validHipCenterYs), 4)
                    if validHipCenterYs else None,
                    "range": round(motionRange, 4),
                },
                "stanceToShoulderRatio": {
                    "min": round(minStanceRatio, 3) if minStanceRatio is not None else None,
                    "max": round(maxStanceRatio, 3) if maxStanceRatio is not None else None,
                    "avg": round(avgStanceRatio, 3) if avgStanceRatio is not None else None,
                },
                "normalizedSymmetryImbalance": {
                    "min": round(minSymmetryScore, 4) if minSymmetryScore is not None else None,
                    "max": round(maxSymmetryScore, 4) if maxSymmetryScore is not None else None,
                    "avg": round(avgSymmetryScore, 4) if avgSymmetryScore is not None else None,
                },
            },
            "repCheckSummary": {
                "stanceIssueCount": stanceIssueCount,
                "symmetryIssueCount": symmetryIssueCount,
                "perRepAvgStanceRatios": [round(value, 3) for value in perRepStanceRatios],
                "perRepMaxSymmetryScores": [round(value, 4) for value in perRepSymmetryScores],
            },
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Front-view squat analysis completed (rep count + stance width + symmetry checks).",
        )

    def _getHipCenterY(self, landmarks: dict) -> Optional[float]:
        leftHip = landmarks.get("left_hip")
        rightHip = landmarks.get("right_hip")
        if leftHip is None or rightHip is None:
            return None
        return (float(leftHip.y) + float(rightHip.y)) / 2.0

    def _getStanceWidthRatio(self, landmarks: dict) -> Optional[float]:
        leftAnkle = landmarks.get("left_ankle")
        rightAnkle = landmarks.get("right_ankle")
        leftShoulder = landmarks.get("left_shoulder")
        rightShoulder = landmarks.get("right_shoulder")

        if (
            leftAnkle is None or rightAnkle is None or
            leftShoulder is None or rightShoulder is None
        ):
            return None

        ankleDistance = abs(rightAnkle.x - leftAnkle.x)
        shoulderDistance = abs(rightShoulder.x - leftShoulder.x)
        if shoulderDistance <= 1e-9:
            return None
        return ankleDistance / shoulderDistance

    def _getFrontSymmetryScore(self, landmarks: dict) -> Optional[float]:
        leftShoulder = landmarks.get("left_shoulder")
        rightShoulder = landmarks.get("right_shoulder")
        leftHip = landmarks.get("left_hip")
        rightHip = landmarks.get("right_hip")

        if (
            leftShoulder is None or rightShoulder is None or
            leftHip is None or rightHip is None
        ):
            return None

        shoulderWidth = abs(rightShoulder.x - leftShoulder.x)
        if shoulderWidth <= 1e-9:
            return None

        shoulderHeightDiff = abs(leftShoulder.y - rightShoulder.y)
        hipHeightDiff = abs(leftHip.y - rightHip.y)
        return ((shoulderHeightDiff / shoulderWidth) + (hipHeightDiff / shoulderWidth)) / 2.0

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
            window = [value for value in values[start:end] if value is not None]
            smoothed.append((sum(window) / len(window)) if window else None)

        return smoothed

    def _detectRepWindows(
        self,
        hipCenterYs: List[Optional[float]],
    ) -> Tuple[int, List[Dict[str, int]], float]:
        validValues = [value for value in hipCenterYs if value is not None]
        if len(validValues) < 20:
            return 0, [], 0.0

        minY = min(validValues)
        maxY = max(validValues)
        motionRange = maxY - minY
        if motionRange < 0.02:
            return 0, [], motionRange

        descendThreshold = minY + (motionRange * 0.28)
        bottomThreshold = minY + (motionRange * 0.62)
        topThreshold = minY + (motionRange * 0.22)

        repWindows: List[Dict[str, int]] = []
        state = "top"
        repStartFrameIndex: Optional[int] = None
        lastBottomFrameIndex: Optional[int] = None

        for frameIndex, hipY in enumerate(hipCenterYs):
            if hipY is None:
                continue

            if state == "top":
                if hipY > descendThreshold:
                    state = "down"
                    repStartFrameIndex = frameIndex

            elif state == "down":
                if hipY > bottomThreshold:
                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                elif hipY <= topThreshold:
                    state = "top"
                    repStartFrameIndex = None
                    lastBottomFrameIndex = None

            elif state == "bottom":
                if lastBottomFrameIndex is None or hipY >= hipCenterYs[lastBottomFrameIndex]:
                    lastBottomFrameIndex = frameIndex
                if hipY < bottomThreshold - (motionRange * 0.08):
                    state = "up"

            elif state == "up":
                if hipY > bottomThreshold:
                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                elif hipY <= topThreshold:
                    if repStartFrameIndex is not None and lastBottomFrameIndex is not None:
                        repLength = frameIndex - repStartFrameIndex
                        if repLength >= 12:
                            repWindows.append(
                                {
                                    "startFrameIndex": repStartFrameIndex,
                                    "bottomFrameIndex": lastBottomFrameIndex,
                                    "endFrameIndex": frameIndex,
                                }
                            )
                    state = "top"
                    repStartFrameIndex = None
                    lastBottomFrameIndex = None

        return len(repWindows), repWindows, motionRange
