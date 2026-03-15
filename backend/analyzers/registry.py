from typing import Dict, Type


# Squat analyzers
try:
    from backend.analyzers.legs.squat.side_squat_analyzer import SideSquatAnalyzer
except Exception:
    SideSquatAnalyzer = None

# Bench press analyzers
try:
    from backend.analyzers.chest.bench_press.front_bench_press_analyzer import FrontBenchPressAnalyzer
except Exception:
    FrontBenchPressAnalyzer = None

try:
    from backend.analyzers.chest.bench_press.side_bench_press_analyzer import SideBenchPressAnalyzer
except Exception:
    SideBenchPressAnalyzer = None


# Pull-up analyzers
try:
    from backend.analyzers.back.pull_up_analyzer import PullUpAnalyzer
except Exception:
    PullUpAnalyzer = None

# Shoulder press analyzers
try:
    from backend.analyzers.shoulders.shoulder_press.front_shoulder_press_analyzer import (
        FrontShoulderPressAnalyzer,
    )
except Exception:
    FrontShoulderPressAnalyzer = None

# Tricep extension analyzers
try:
    from backend.analyzers.arms.tricep_extension.side_tricep_extension_analyzer import (
        SideTricepExtensionAnalyzer,
    )
except Exception:
    SideTricepExtensionAnalyzer = None

# Bicep curl analyzers
try:
    from backend.analyzers.arms.bicep_curl.side_bicep_curl_analyzer import (
        SideBicepCurlAnalyzer,
    )
except Exception:
    SideBicepCurlAnalyzer = None


def _buildRegistry() -> Dict[str, Type]:
    registry: Dict[str, Type] = {}

    if SideSquatAnalyzer is not None:
        registry["squat_side"] = SideSquatAnalyzer

    if FrontBenchPressAnalyzer is not None:
        registry["bench_front"] = FrontBenchPressAnalyzer

    if SideBenchPressAnalyzer is not None:
        registry["bench_side"] = SideBenchPressAnalyzer

    if PullUpAnalyzer is not None:
        registry["pullup_back"] = PullUpAnalyzer
        
    if FrontShoulderPressAnalyzer is not None:
        registry["shoulder_press_front"] = FrontShoulderPressAnalyzer

    if SideTricepExtensionAnalyzer is not None:
        registry["tricep_extension_side"] = SideTricepExtensionAnalyzer

    if SideBicepCurlAnalyzer is not None:
        registry["bicep_curl_side"] = SideBicepCurlAnalyzer

    return registry


ANALYZER_REGISTRY = _buildRegistry()


def getAnalyzer(exerciseName: str):
    """
    Returns an analyzer instance for the given exercise name.
    Raises ValueError if not supported (or analyzer file not implemented yet).
    """
    analyzerClass = ANALYZER_REGISTRY.get(exerciseName)

    if analyzerClass is None:
        supportedExercises = ", ".join(sorted(ANALYZER_REGISTRY.keys())) or "(none yet)"
        raise ValueError(
            f"Unsupported exercise '{exerciseName}'. Supported exercises: {supportedExercises}"
        )

    return analyzerClass()


def getSupportedExercises():
    return sorted(ANALYZER_REGISTRY.keys())
