from typing import Any, Dict, List, Optional, Tuple

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class FrontSquatAnalyzer(BaseAnalyzer):
    """
    V1.0 front-view squat analyzer:
    - Rep count (using average left/right knee angle state machine)
    - Stance-width check (ankle distance relative to shoulder width)
    - Left/right symmetry check (shoulder + hip height balance)

    Notes:
    - This is intentionally lighter than the side-view analyzer.
    - Front view is used here as a supplemental checker.
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

        # Tunable thresholds
        topThreshold = float(options.get("topThreshold", 155.0))
        bottomThreshold = float(options.get("bottomThreshold", 115.0))

        stanceTooNarrowRatio = float(options.get("stanceTooNarrowRatio", 0.9))
        stanceTooWideRatio = float(options.get("stanceTooWideRatio", 1.8))

        mildAsymmetryThreshold = float(options.get("mildAsymmetryThreshold", 0.08))
        highAsymmetryThreshold = float(options.get("highAsymmetryThreshold", 0.14))

        # Signals across all frames
        avgKneeAngles: List[Optional[float]] = []
        stanceRatios: List[Optional[float]] = []
        symmetryScores: List[Optional[float]] = []

        for frame in poseFrames:
            if not frame.hasPose:
                avgKneeAngles.append(None)
                stanceRatios.append(None)
                symmetryScores.append(None)
                continue

            landmarks = frame.landmarks

            avgKneeAngle = self._getAverageKneeAngle(landmarks)
            stanceRatio = self._getStanceWidthRatio(landmarks)
            symmetryScore = self._getFrontSymmetryScore(landmarks)

            avgKneeAngles.append(avgKneeAngle)
            stanceRatios.append(stanceRatio)
            symmetryScores.append(symmetryScore)

        validKneeAngles = [v for v in avgKneeAngles if v is not None]
        validStanceRatios = [v for v in stanceRatios if v is not None]
        validSymmetryScores = [v for v in symmetryScores if v is not None]

        totalFrames = len(poseFrames)
        validKneeAngleFrames = len(validKneeAngles)
        validStanceFrames = len(validStanceRatios)
        validSymmetryFrames = len(validSymmetryScores)

        if validKneeAngleFrames < 20:
            return self.buildFailedResult(
                message="Not enough valid knee-angle frames to analyze front-view squat.",
                warnings=["Try a clearer front-view video with full body visible."],
            )

        # Smooth signals a bit
        smoothedKneeAngles = self._movingAverageOptional(avgKneeAngles, windowSize=5)
        smoothedStanceRatios = self._movingAverageOptional(stanceRatios, windowSize=5)
        smoothedSymmetryScores = self._movingAverageOptional(symmetryScores, windowSize=5)

        # Rep detection
        repCount, repWindows = self._detectSquatRepWindows(
            smoothedKneeAngles,
            topThreshold=topThreshold,
            bottomThreshold=bottomThreshold,
        )

        issues: List[Dict[str, Any]] = []
        warnings: List[str] = []
        repFeedback: List[Dict[str, Any]] = []

        validKneeRatio = validKneeAngleFrames / totalFrames if totalFrames > 0 else 0.0
        validStanceRatio = validStanceFrames / totalFrames if totalFrames > 0 else 0.0
        validSymmetryRatio = validSymmetryFrames / totalFrames if totalFrames > 0 else 0.0

        if validKneeRatio < 0.7:
            warnings.append(
                "Front-view knee-angle tracking quality was low for many frames; rep detection may be less reliable."
            )

        if validStanceRatio < 0.6:
            warnings.append(
                "Foot/shoulder tracking quality was low for many frames; stance-width feedback may be less reliable."
            )

        if validSymmetryRatio < 0.6:
            warnings.append(
                "Left/right body tracking quality was low for many frames; symmetry feedback may be less reliable."
            )

        perRepAvgStanceRatios: List[float] = []
        perRepMaxSymmetryScores: List[float] = []

        stanceIssueCount = 0
        symmetryIssueCount = 0

        for repIndex, repWindow in enumerate(repWindows, start=1):
            repIssuesCodes: List[str] = []
            repQualityPenalties = 0.0

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
                sum(repStanceWindow) / len(repStanceWindow)
                if repStanceWindow else None
            )
            repMaxSymmetryScore = max(repSymmetryWindow) if repSymmetryWindow else None

            if repAvgStanceRatio is not None:
                perRepAvgStanceRatios.append(repAvgStanceRatio)

            if repMaxSymmetryScore is not None:
                perRepMaxSymmetryScores.append(repMaxSymmetryScore)

            # --- Stance width (per rep average) ---
            if repAvgStanceRatio is not None:
                if repAvgStanceRatio < stanceTooNarrowRatio:
                    repIssuesCodes.append("stance_too_narrow")
                    repQualityPenalties += 12
                    stanceIssueCount += 1
                elif repAvgStanceRatio > stanceTooWideRatio:
                    repIssuesCodes.append("stance_too_wide")
                    repQualityPenalties += 10
                    stanceIssueCount += 1

            # --- Symmetry (per rep max imbalance) ---
            if repMaxSymmetryScore is not None:
                if repMaxSymmetryScore >= highAsymmetryThreshold:
                    repIssuesCodes.append("symmetry_high_imbalance")
                    repQualityPenalties += 15
                    symmetryIssueCount += 1
                elif repMaxSymmetryScore >= mildAsymmetryThreshold:
                    repIssuesCodes.append("symmetry_mild_imbalance")
                    repQualityPenalties += 8
                    symmetryIssueCount += 1

            repQuality = max(0.0, min(100.0, 100.0 - repQualityPenalties))

            repFeedback.append(
                {
                    "rep": repIndex,
                    "quality": round(repQuality, 1),
                    "issues": repIssuesCodes,
                    "startFrameIndex": startFrame,
                    "bottomFrameIndex": bottomFrame,
                    "endFrameIndex": endFrame,
                    "checks": {
                        "stanceWidth": {
                            "avgStanceToShoulderRatio": round(repAvgStanceRatio, 3)
                            if repAvgStanceRatio is not None else None
                        },
                        "symmetry": {
                            "maxNormalizedImbalance": round(repMaxSymmetryScore, 4)
                            if repMaxSymmetryScore is not None else None
                        },
                    },
                }
            )

        # Top-level summarized issues
        if repCount > 0:
            if stanceIssueCount > 0:
                severity = "medium" if stanceIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="stance_width_issue",
                        message=f"Stance width looked suboptimal in {stanceIssueCount}/{repCount} rep(s).",
                        severity=severity,
                    )
                )

            if symmetryIssueCount > 0:
                severity = "medium" if symmetryIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="asymmetry_issue",
                        message=f"Left/right balance looked uneven in {symmetryIssueCount}/{repCount} rep(s).",
                        severity=severity,
                    )
                )

        if repCount == 0:
            warnings.append(
                "No front-view squat reps were confidently detected. Try a clearer front view and full-body framing."
            )

        # Summary score = average rep quality if reps exist
        summaryScore = None
        repQualities = [rep["quality"] for rep in repFeedback if rep.get("quality") is not None]
        if repQualities:
            summaryScore = round(sum(repQualities) / len(repQualities), 1)

        # Overall metrics
        minKneeAngle = min(validKneeAngles) if validKneeAngles else None
        maxKneeAngle = max(validKneeAngles) if validKneeAngles else None
        avgKneeAngle = (sum(validKneeAngles) / len(validKneeAngles)) if validKneeAngles else None

        minStanceRatio = min(validStanceRatios) if validStanceRatios else None
        maxStanceRatio = max(validStanceRatios) if validStanceRatios else None
        avgStanceRatio = (sum(validStanceRatios) / len(validStanceRatios)) if validStanceRatios else None

        minSymmetryScore = min(validSymmetryScores) if validSymmetryScores else None
        maxSymmetryScore = max(validSymmetryScores) if validSymmetryScores else None
        avgSymmetryScore = (sum(validSymmetryScores) / len(validSymmetryScores)) if validSymmetryScores else None

        metrics = {
            "view": "front",
            "signalQuality": {
                "totalFrames": totalFrames,
                "validKneeAngleFrames": validKneeAngleFrames,
                "validKneeAngleRatio": round(validKneeRatio, 4),
                "validStanceFrames": validStanceFrames,
                "validStanceRatio": round(validStanceRatio, 4),
                "validSymmetryFrames": validSymmetryFrames,
                "validSymmetryRatio": round(validSymmetryRatio, 4),
            },
            "globalSignals": {
                "avgLeftRightKneeAngleDeg": {
                    "min": round(minKneeAngle, 2) if minKneeAngle is not None else None,
                    "max": round(maxKneeAngle, 2) if maxKneeAngle is not None else None,
                    "avg": round(avgKneeAngle, 2) if avgKneeAngle is not None else None,
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
                "perRepAvgStanceToShoulderRatio": [round(v, 3) for v in perRepAvgStanceRatios],
                "perRepMaxNormalizedImbalance": [round(v, 4) for v in perRepMaxSymmetryScores],
            },
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Front-view squat analysis completed (stance width + symmetry checks).",
        )

    def _getAverageKneeAngle(self, landmarks: dict) -> Optional[float]:
        """
        Average of left and right knee angles when available.
        Falls back to whichever side is available.
        """
        leftKneeAngle = getLandmarkAngle(landmarks, "left_hip", "left_knee", "left_ankle")
        rightKneeAngle = getLandmarkAngle(landmarks, "right_hip", "right_knee", "right_ankle")

        available = [v for v in [leftKneeAngle, rightKneeAngle] if v is not None]
        if not available:
            return None

        return sum(available) / len(available)

    def _getStanceWidthRatio(self, landmarks: dict) -> Optional[float]:
        """
        Returns ankle distance / shoulder distance.
        This is a scale-normalized way to estimate stance width from the front.
        """
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
        """
        Returns a simple normalized imbalance score using left/right shoulder and hip height.
        Lower is better, higher means more visible left/right asymmetry.

        Normalization uses shoulder width so the score is less sensitive to video scale.
        """
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

        # Average the two normalized left/right height imbalances
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
            window = [v for v in values[start:end] if v is not None]

            if not window:
                smoothed.append(None)
            else:
                smoothed.append(sum(window) / len(window))

        return smoothed

    def _detectSquatRepWindows(
        self,
        kneeAngles: List[Optional[float]],
        topThreshold: float = 155.0,
        bottomThreshold: float = 115.0,
    ) -> Tuple[int, List[Dict[str, int]]]:
        """
        Detect squat reps and return rep windows:
        each rep window includes startFrameIndex, bottomFrameIndex, endFrameIndex

        Uses the same basic state-machine pattern as the side analyzer,
        but on front-view average knee angle.
        """
        repWindows: List[Dict[str, int]] = []

        state = "top"
        repStartFrameIndex: Optional[int] = None
        lastBottomFrameIndex: Optional[int] = None
        lastBottomAngle: Optional[float] = None

        for frameIndex, angle in enumerate(kneeAngles):
            if angle is None:
                continue

            if state == "top":
                if angle < topThreshold:
                    state = "down"
                    repStartFrameIndex = frameIndex
                    lastBottomFrameIndex = None
                    lastBottomAngle = None

            elif state == "down":
                if angle < bottomThreshold:
                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                    lastBottomAngle = angle
                elif angle >= topThreshold:
                    # Noise / partial movement returned to top
                    state = "top"
                    repStartFrameIndex = None
                    lastBottomFrameIndex = None
                    lastBottomAngle = None

            elif state == "bottom":
                # Keep the deepest frame as bottom frame
                if lastBottomAngle is None or angle < lastBottomAngle:
                    lastBottomAngle = angle
                    lastBottomFrameIndex = frameIndex

                if angle > bottomThreshold + 5:
                    state = "up"

            elif state == "up":
                if angle < bottomThreshold:
                    state = "bottom"
                    if lastBottomAngle is None or angle < lastBottomAngle:
                        lastBottomAngle = angle
                        lastBottomFrameIndex = frameIndex
                elif angle >= topThreshold:
                    if repStartFrameIndex is not None and lastBottomFrameIndex is not None:
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
                    lastBottomAngle = None

        return len(repWindows), repWindows