from typing import Any, Dict, List, Optional, Tuple
import math

from backend.analyzers.base_analyzer import BaseAnalyzer
from backend.core.biomechanics.angles import getLandmarkAngle


class SideSquatAnalyzer(BaseAnalyzer):
    """
    V1.1 side-view squat analyzer:
    - Rep count (knee angle state machine)
    - Depth check (per-rep bottom knee angle)
    - Torso lean check (per-rep max torso lean from vertical)
    - Lockout check (per-rep top knee angle after ascent)
    """

    def __init__(self):
        super().__init__("squat")
        self._frameWidth = 1.0
        self._frameHeight = 1.0

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
        sidePreference = options.get("side", "left")  # "left" or "right"
        self._frameWidth = float((videoMetadata or {}).get("width") or 1.0)
        self._frameHeight = float((videoMetadata or {}).get("height") or 1.0)

        # Signals across all frames
        kneeAngles: List[Optional[float]] = []
        torsoLeanAngles: List[Optional[float]] = []  # angle from vertical in degrees (higher = more lean)

        for frame in poseFrames:
            if not frame.hasPose:
                kneeAngles.append(None)
                torsoLeanAngles.append(None)
                continue

            kneeAngle = self._getPreferredKneeAngle(frame.landmarks, sidePreference)
            torsoLean = self._getPreferredTorsoLeanFromVertical(frame.landmarks, sidePreference)

            kneeAngles.append(kneeAngle)
            torsoLeanAngles.append(torsoLean)

        validKneeAngles = [a for a in kneeAngles if a is not None]
        validTorsoLeans = [a for a in torsoLeanAngles if a is not None]

        totalFrames = len(poseFrames)
        validKneeAngleFrames = len(validKneeAngles)
        validTorsoLeanFrames = len(validTorsoLeans)

        if validKneeAngleFrames < 20:
            return self.buildFailedResult(
                message="Not enough valid knee-angle frames to analyze side-view squat.",
                warnings=["Try a clearer side-view video with full body visible."],
            )

        # Smooth signals slightly
        smoothedKneeAngles = self._movingAverageOptional(kneeAngles, windowSize=5)
        smoothedTorsoLeanAngles = self._movingAverageOptional(torsoLeanAngles, windowSize=5)

        # Rep detection + events
        repCount, repWindows = self._detectSquatRepWindows(smoothedKneeAngles)

        issues: List[Dict[str, Any]] = []
        warnings: List[str] = []
        repFeedback: List[Dict[str, Any]] = []

        validKneeRatio = validKneeAngleFrames / totalFrames if totalFrames > 0 else 0.0
        validTorsoRatio = validTorsoLeanFrames / totalFrames if totalFrames > 0 else 0.0

        if validKneeRatio < 0.7:
            warnings.append(
                "Knee tracking quality was low for many frames; squat feedback may be less reliable."
            )

        if validTorsoRatio < 0.6:
            warnings.append(
                "Torso tracking quality was low for many frames; torso lean feedback may be less reliable."
            )

        # Per-rep checks (depth, torso lean, lockout)
        perRepBottomKneeAngles: List[float] = []
        perRepMaxTorsoLeanAngles: List[float] = []
        perRepTopKneeAnglesAfterAscent: List[float] = []
        visualEvents: List[Dict[str, Any]] = []

        depthIssueCount = 0
        torsoIssueCount = 0
        lockoutIssueCount = 0

        for repIndex, repWindow in enumerate(repWindows, start=1):
            repIssuesCodes: List[str] = []
            repQualityPenalties = 0.0

            startFrame = repWindow["startFrameIndex"]
            bottomFrame = repWindow["bottomFrameIndex"]
            endFrame = repWindow["endFrameIndex"]

            repKneeWindow = [
                value for value in smoothedKneeAngles[startFrame:endFrame + 1] if value is not None
            ]
            repTorsoWindow = [
                value for value in smoothedTorsoLeanAngles[startFrame:endFrame + 1] if value is not None
            ]
            ascentKneeWindow = [
                value for value in smoothedKneeAngles[bottomFrame:endFrame + 1] if value is not None
            ]

            repBottomKneeAngle = min(repKneeWindow) if repKneeWindow else None
            repMaxTorsoLean = max(repTorsoWindow) if repTorsoWindow else None
            repTopKneeAngleAfterAscent = max(ascentKneeWindow) if ascentKneeWindow else None

            if repBottomKneeAngle is not None:
                perRepBottomKneeAngles.append(repBottomKneeAngle)

            if repMaxTorsoLean is not None:
                perRepMaxTorsoLeanAngles.append(repMaxTorsoLean)

            if repTopKneeAngleAfterAscent is not None:
                perRepTopKneeAnglesAfterAscent.append(repTopKneeAngleAfterAscent)

            # --- Depth (per rep) ---
            # Lower bottom knee angle generally means deeper squat.
            # These thresholds are rough and should be tuned with your data.
            if repBottomKneeAngle is not None:
                if repBottomKneeAngle > 100:
                    repIssuesCodes.append("depth_high")
                    repQualityPenalties += 20
                    depthIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "depth_high",
                            "message": "Not deep enough in this rep.",
                            "severity": "high",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                            "measuredAngleDeg": round(repBottomKneeAngle, 2),
                            "targetAngleDeg": 85.0,
                            "joint": "knee",
                        }
                    )
                elif repBottomKneeAngle > 85:
                    repIssuesCodes.append("depth_moderate")
                    repQualityPenalties += 10
                    depthIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "depth_moderate",
                            "message": "Depth was a bit shallow in this rep.",
                            "severity": "medium",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                            "measuredAngleDeg": round(repBottomKneeAngle, 2),
                            "targetAngleDeg": 85.0,
                            "joint": "knee",
                        }
                    )

            # --- Torso lean (per rep max lean from vertical) ---
            # Angle from vertical: higher means more forward lean.
            # This is a heuristic and should be tuned.
            if repMaxTorsoLean is not None:
                if repMaxTorsoLean > 55:
                    repIssuesCodes.append("torso_lean_excessive")
                    repQualityPenalties += 15
                    torsoIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "torso_lean_excessive",
                            "message": "Too much forward torso lean.",
                            "severity": "high",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                            "measuredAngleDeg": round(repMaxTorsoLean, 2),
                            "targetAngleDeg": 45.0,
                            "joint": "torso",
                        }
                    )
                elif repMaxTorsoLean > 45:
                    repIssuesCodes.append("torso_lean_moderate")
                    repQualityPenalties += 8
                    torsoIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "torso_lean_moderate",
                            "message": "Moderate forward torso lean.",
                            "severity": "medium",
                            "frameIndex": bottomFrame,
                            "rep": repIndex,
                            "measuredAngleDeg": round(repMaxTorsoLean, 2),
                            "targetAngleDeg": 45.0,
                            "joint": "torso",
                        }
                    )

            # --- Lockout at top (after ascent) ---
            # If they never get close to straight knees near the top, may be incomplete lockout.
            if repTopKneeAngleAfterAscent is not None:
                if repTopKneeAngleAfterAscent < 155:
                    repIssuesCodes.append("lockout_incomplete")
                    repQualityPenalties += 12
                    lockoutIssueCount += 1
                    visualEvents.append(
                        {
                            "type": "lockout_incomplete",
                            "message": "Incomplete knee lockout at top.",
                            "severity": "medium",
                            "frameIndex": endFrame,
                            "rep": repIndex,
                            "measuredAngleDeg": round(repTopKneeAngleAfterAscent, 2),
                            "targetAngleDeg": 155.0,
                            "joint": "knee",
                        }
                    )

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
                        "depth": {
                            "bottomKneeAngleDeg": round(repBottomKneeAngle, 2)
                            if repBottomKneeAngle is not None else None
                        },
                        "torsoLean": {
                            "maxTorsoLeanFromVerticalDeg": round(repMaxTorsoLean, 2)
                            if repMaxTorsoLean is not None else None
                        },
                        "lockout": {
                            "topKneeAngleAfterAscentDeg": round(repTopKneeAngleAfterAscent, 2)
                            if repTopKneeAngleAfterAscent is not None else None
                        },
                    },
                }
            )

        # Top-level issues summarized from per-rep issues
        if repCount > 0:
            if depthIssueCount > 0:
                severity = "medium" if depthIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="depth_consistency_issue",
                        message=f"Depth looked limited in {depthIssueCount}/{repCount} rep(s).",
                        severity=severity,
                    )
                )

            if torsoIssueCount > 0:
                severity = "medium" if torsoIssueCount >= max(1, repCount // 2) else "low"
                issues.append(
                    self.buildIssue(
                        code="torso_lean_issue",
                        message=f"Forward torso lean was elevated in {torsoIssueCount}/{repCount} rep(s).",
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

        if repCount == 0:
            warnings.append(
                "No side-view squat reps were confidently detected. Try a clearer side view and full-body framing."
            )

        # Summary score = average rep quality if reps exist
        summaryScore = None
        repQualities = [rep["quality"] for rep in repFeedback if rep.get("quality") is not None]
        if repQualities:
            summaryScore = round(sum(repQualities) / len(repQualities), 1)

        # Overall metrics (clear/debuggable)
        minKneeAngle = min(validKneeAngles) if validKneeAngles else None
        maxKneeAngle = max(validKneeAngles) if validKneeAngles else None
        avgKneeAngle = (sum(validKneeAngles) / len(validKneeAngles)) if validKneeAngles else None

        minTorsoLean = min(validTorsoLeans) if validTorsoLeans else None
        maxTorsoLean = max(validTorsoLeans) if validTorsoLeans else None
        avgTorsoLean = (sum(validTorsoLeans) / len(validTorsoLeans)) if validTorsoLeans else None

        metrics = {
            "view": "side",
            "visualFeedback": {
                "events": visualEvents,
            },
            "signalQuality": {
                "totalFrames": totalFrames,
                "validKneeAngleFrames": validKneeAngleFrames,
                "validKneeAngleRatio": round(validKneeRatio, 4),
                "validTorsoLeanFrames": validTorsoLeanFrames,
                "validTorsoLeanRatio": round(validTorsoRatio, 4),
            },
            "globalSignals": {
                "kneeAngleDeg": {
                    "min": round(minKneeAngle, 2) if minKneeAngle is not None else None,
                    "max": round(maxKneeAngle, 2) if maxKneeAngle is not None else None,
                    "avg": round(avgKneeAngle, 2) if avgKneeAngle is not None else None,
                },
                "torsoLeanFromVerticalDeg": {
                    "min": round(minTorsoLean, 2) if minTorsoLean is not None else None,
                    "max": round(maxTorsoLean, 2) if maxTorsoLean is not None else None,
                    "avg": round(avgTorsoLean, 2) if avgTorsoLean is not None else None,
                },
            },
            "repCheckSummary": {
                "depthIssueCount": depthIssueCount,
                "torsoLeanIssueCount": torsoIssueCount,
                "lockoutIssueCount": lockoutIssueCount,
                "perRepBottomKneeAnglesDeg": [round(v, 2) for v in perRepBottomKneeAngles],
                "perRepMaxTorsoLeanFromVerticalDeg": [round(v, 2) for v in perRepMaxTorsoLeanAngles],
                "perRepTopKneeAnglesAfterAscentDeg": [round(v, 2) for v in perRepTopKneeAnglesAfterAscent],
            },
        }

        return self.buildSuccessResult(
            repCount=repCount,
            summaryScore=summaryScore,
            issues=issues,
            repFeedback=repFeedback,
            metrics=metrics,
            warnings=warnings,
            message="Side-view squat analysis completed (depth + torso lean + lockout checks).",
        )

    def _getPreferredKneeAngle(self, landmarks: dict, sidePreference: str) -> Optional[float]:
        """
        Knee angle = angle(hip, knee, ankle)
        Chooses preferred side first, then falls back.
        """
        if sidePreference == "right":
            primary = ("right_hip", "right_knee", "right_ankle")
            fallback = ("left_hip", "left_knee", "left_ankle")
        else:
            primary = ("left_hip", "left_knee", "left_ankle")
            fallback = ("right_hip", "right_knee", "right_ankle")

        angle = getLandmarkAngle(landmarks, *primary)
        if angle is not None:
            return angle
        return getLandmarkAngle(landmarks, *fallback)

    def _getPreferredTorsoLeanFromVertical(self, landmarks: dict, sidePreference: str) -> Optional[float]:
        """
        Returns torso lean angle from vertical (in degrees).
        Uses shoulder and hip on preferred side (fallback to other side).
        0 = perfectly vertical torso line, larger values = more lean.
        """
        if sidePreference == "right":
            primary = ("right_shoulder", "right_hip")
            fallback = ("left_shoulder", "left_hip")
        else:
            primary = ("left_shoulder", "left_hip")
            fallback = ("right_shoulder", "right_hip")

        lean = self._torsoLeanFromVertical(landmarks, *primary)
        if lean is not None:
            return lean
        return self._torsoLeanFromVertical(landmarks, *fallback)

    def _torsoLeanFromVertical(self, landmarks: dict, shoulderName: str, hipName: str) -> Optional[float]:
        shoulder = landmarks.get(shoulderName)
        hip = landmarks.get(hipName)

        if shoulder is None or hip is None:
            return None

        # Landmarks are normalized independently by frame width/height.
        # Convert to pixel-space deltas so the angle is not skewed by aspect ratio.
        dx = (shoulder.x - hip.x) * self._frameWidth
        dy = (shoulder.y - hip.y) * self._frameHeight

        # If both are same point / unusable
        if dx == 0 and dy == 0:
            return None

        # Angle from vertical axis:
        # atan2(|dx|, |dy|) => 0 when vertical, increases as torso leans
        leanRad = math.atan2(abs(dx), abs(dy) if abs(dy) > 1e-9 else 1e-9)
        leanDeg = math.degrees(leanRad)

        return leanDeg

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

    def _detectSquatRepWindows(self, kneeAngles: List[Optional[float]]) -> Tuple[int, List[Dict[str, int]]]:
        """
        Detect squat reps and return rep windows:
        each rep window includes startFrameIndex, bottomFrameIndex, endFrameIndex

        V1 state machine based on knee angle.
        """
        repWindows: List[Dict[str, int]] = []

        topThreshold = 155.0
        bottomThreshold = 105.0

        state = "top"
        repStartFrameIndex: Optional[int] = None
        lastBottomFrameIndex: Optional[int] = None

        for frameIndex, angle in enumerate(kneeAngles):
            if angle is None:
                continue

            if state == "top":
                if angle < topThreshold:
                    state = "down"
                    repStartFrameIndex = frameIndex

            elif state == "down":
                if angle < bottomThreshold:
                    state = "bottom"
                    lastBottomFrameIndex = frameIndex
                elif angle >= topThreshold:
                    # Noise / partial movement returned to top
                    state = "top"
                    repStartFrameIndex = None
                    lastBottomFrameIndex = None

            elif state == "bottom":
                # Track deepest point continuously while in bottom
                if lastBottomFrameIndex is None or (angle is not None):
                    if lastBottomFrameIndex is None:
                        lastBottomFrameIndex = frameIndex
                if angle > bottomThreshold + 5:
                    state = "up"

            elif state == "up":
                # Update bottom if user dips again before finishing the rep
                if angle < bottomThreshold:
                    state = "bottom"
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

        return len(repWindows), repWindows
