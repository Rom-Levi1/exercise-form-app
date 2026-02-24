from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAnalyzer(ABC):
    """
    All exercise analyzers should inherit from this class and return the same result structure.
    """

    def __init__(self, exerciseName: str):
        self.exerciseName = exerciseName

    @abstractmethod
    def analyze(
        self,
        videoPath: str,
        poseFrames: List[Any],
        videoMetadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Must return a shared result dict.
        """
        raise NotImplementedError

    def buildSuccessResult(
        self,
        repCount: int = 0,
        summaryScore: Optional[float] = None,
        issues: Optional[List[Dict[str, Any]]] = None,
        repFeedback: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        message: str = "Analysis completed successfully.",
    ) -> Dict[str, Any]:
        return {
            "exercise": self.exerciseName,
            "status": "success",
            "message": message,
            "repCount": repCount,
            "summaryScore": summaryScore,
            "issues": issues or [],
            "repFeedback": repFeedback or [],
            "metrics": metrics or {},
            "warnings": warnings or [],
        }

    def buildFailedResult(
        self,
        message: str = "Analysis failed.",
        warnings: Optional[List[str]] = None,
        issues: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "exercise": self.exerciseName,
            "status": "failed",
            "message": message,
            "repCount": 0,
            "summaryScore": None,
            "issues": issues or [],
            "repFeedback": [],
            "metrics": {},
            "warnings": warnings or [],
        }

    def buildIssue(
        self,
        code: str,
        message: str,
        severity: str = "low",
    ) -> Dict[str, str]:
        return {
            "code": code,
            "message": message,
            "severity": severity,  # "low" | "medium" | "high"
        }