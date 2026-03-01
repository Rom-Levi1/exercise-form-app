from typing import Any, Dict, List, Optional

from backend.analyzers.base_analyzer import BaseAnalyzer


class FrontSquatAnalyzer(BaseAnalyzer):
    """
    V1.1 front-view squat analyzer (supplemental only):
    - No rep counting
    - Stance-width check (ankle distance relative to shoulder width)
    - Left/right symmetry check (shoulder + hip height balance)

    Notes:
    - Front view is treated as a clip-level checker, not a rep detector.
    - Main squat rep counting and primary form checks should come from the side view.
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
        stanceTooNarrowRatio = float(options.get("stanceTooNarrowRatio", 0.9))
        stanceTooWideRatio = float(options.get("stanceTooWideRatio", 1.8))

        mildAsymmetryThreshold = float(options.get("mildAsymmetryThreshold", 0.08))
        highAsymmetryThreshold = float(options.get("highAsymmetryThreshold", 0.14))

        stanceRatios: List[Optional[float]] = []
        symmetryScores: List[Optional[float]] = []

        for frame in poseFrames:
            if not frame.hasPose:
                stanceRatios.append(None)
                symmetryScores.append(None)
                continue

            landmarks = frame.landmarks

            stanceRatio = self._getStanceWidthRatio(landmarks)
            symmetryScore = self._getFrontSymmetryScore(landmarks)

            stanceRatios.append(stanceRatio)
            symmetryScores.append(symmetryScore)

        validStanceRatios = [v for v in stanceRatios if v is not None]
        validSymmetryScores = [v for v in symmetryScores if v is not None]

        totalFrames = len(poseFrames)
        validStanceFrames = len(validStanceRatios)
        validSymmetryFrames = len(validSymmetryScores)

        if validStanceFrames < 20 and validSymmetryFrames < 20:
            return self.buildFailedResult(
                message="Not enough valid front-view frames to analyze squat stance/symmetry.",
                warnings=["Try a clearer front-view video with full body visible."],
            )

        # Smooth signals slightly
        smoothedStanceRatios = self._movingAverageOptional(stanceRatios, windowSize=5)
        smoothedSymmetryScores = self._movingAverageOptional(symmetryScores, windowSize=5)

        validSmoothedStanceRatios = [v for v in smoothedStanceRatios if v is not None]
        validSmoothedSymmetryScores = [v for v in smoothedSymmetryScores if v is not None]

        issues: List[Dict[str, Any]] = []
        warnings: List[str] = []

        validStanceRatio = validStanceFrames / totalFrames if totalFrames > 0 else 0.0
        validSymmetryRatio = validSymmetryFrames / totalFrames if totalFrames > 0 else 0.0

        if validStanceRatio < 0.6:
            warnings.append(
                "Foot/shoulder tracking quality was low for many frames; stance-width feedback may be less reliable."
            )

        if validSymmetryRatio < 0.6:
            warnings.append(
                "Left/right body tracking quality was low for many frames; symmetry feedback may be less reliable."
            )

        # -----------------------------
        # Clip-level stance evaluation
        # -----------------------------
        avgStanceRatio = (
            sum(validSmoothedStanceRatios) / len(validSmoothedStanceRatios)
            if validSmoothedStanceRatios else None
        )

        minStanceRatio = min(validSmoothedStanceRatios) if validSmoothedStanceRatios else None
        maxStanceRatio = max(validSmoothedStanceRatios) if validSmoothedStanceRatios else None

        stanceClassification = "unknown"
        stanceQualityPenalty = 0.0

        if avgStanceRatio is not None:
            if avgStanceRatio < stanceTooNarrowRatio:
                stanceClassification = "too_narrow"
                stanceQualityPenalty += 12
                issues.append(
                    self.buildIssue(
                        code="stance_width_issue",
                        message="Overall stance width looked too narrow for much of the clip.",
                        severity="medium",
                    )
                )
            elif avgStanceRatio > stanceTooWideRatio:
                stanceClassification = "too_wide"
                stanceQualityPenalty += 10
                issues.append(
                    self.buildIssue(
                        code="stance_width_issue",
                        message="Overall stance width looked very wide for much of the clip.",
                        severity="low",
                    )
                )
            else:
                stanceClassification = "acceptable"

        # -----------------------------
        # Clip-level symmetry evaluation
        # -----------------------------
        avgSymmetryScore = (
            sum(validSmoothedSymmetryScores) / len(validSmoothedSymmetryScores)
            if validSmoothedSymmetryScores else None
        )

        maxSymmetryScore = max(validSmoothedSymmetryScores) if validSmoothedSymmetryScores else None
        minSymmetryScore = min(validSmoothedSymmetryScores) if validSmoothedSymmetryScores else None

        symmetryClassification = "unknown"
        symmetryQualityPenalty = 0.0

        if maxSymmetryScore is not None:
            if maxSymmetryScore >= highAsymmetryThreshold:
                symmetryClassification = "high_imbalance"
                symmetryQualityPenalty += 15
                issues.append(
                    self.buildIssue(
                        code="asymmetry_issue",
                        message="Clear left/right imbalance appeared during parts of the clip.",
                        severity="medium",
                    )
                )
            elif maxSymmetryScore >= mildAsymmetryThreshold:
                symmetryClassification = "mild_imbalance"
                symmetryQualityPenalty += 8
                issues.append(
                    self.buildIssue(
                        code="asymmetry_issue",
                        message="Mild left/right imbalance appeared during parts of the clip.",
                        severity="low",
                    )
                )
            else:
                symmetryClassification = "balanced"

        # Clip-level summary score
        qualityPenalty = stanceQualityPenalty + symmetryQualityPenalty
        summaryScore = max(0.0, min(100.0, 100.0 - qualityPenalty))
        summaryScore = round(summaryScore, 1)

        # Put clip-level checks in repFeedback[0] to preserve familiar structure,
        # but mark it explicitly as a clip-level summary (not a rep).
        repFeedback = [
            {
                "rep": None,
                "quality": summaryScore,
                "issues": [issue.get("code") for issue in issues],
                "checks": {
                    "stanceWidth": {
                        "classification": stanceClassification,
                        "avgStanceToShoulderRatio": round(avgStanceRatio, 3)
                        if avgStanceRatio is not None else None,
                        "minStanceToShoulderRatio": round(minStanceRatio, 3)
                        if minStanceRatio is not None else None,
                        "maxStanceToShoulderRatio": round(maxStanceRatio, 3)
                        if maxStanceRatio is not None else None,
                    },
                    "symmetry": {
                        "classification": symmetryClassification,
                        "avgNormalizedImbalance": round(avgSymmetryScore, 4)
                        if avgSymmetryScore is not None else None,
                        "maxNormalizedImbalance": round(maxSymmetryScore, 4)
                        if maxSymmetryScore is not None else None,
                        "minNormalizedImbalance": round(minSymmetryScore, 4)
                        if minSymmetryScore is not None else None,
                    },
                },
            }
        ]

        warnings.append(
            "Front view is used for supplemental squat checks only and does not estimate reps."
        )

        metrics = {
            "view": "front",
            "analysisMode": "clip_level_only",
            "signalQuality": {
                "totalFrames": totalFrames,
                "validStanceFrames": validStanceFrames,
                "validStanceRatio": round(validStanceRatio, 4),
                "validSymmetryFrames": validSymmetryFrames,
                "validSymmetryRatio": round(validSymmetryRatio, 4),
            },
            "globalSignals": {
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
            "clipCheckSummary": {
                "stanceClassification": stanceClassification,
                "symmetryClassification": symmetryClassification,
            },
        }

        return self.buildSuccessResult(
            repCount=0,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Front-view squat analysis completed (stance width + symmetry checks only).",
        )

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