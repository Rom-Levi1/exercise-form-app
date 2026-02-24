from typing import Dict, Type


# Lazy-ish imports with fallback so the registry file can exist before all analyzers are implemented
try:
    from backend.analyzers.legs.squat_analyzer import SquatAnalyzer
except Exception:
    SquatAnalyzer = None

try:
    from backend.analyzers.legs.lunge_analyzer import LungeAnalyzer
except Exception:
    LungeAnalyzer = None

try:
    from backend.analyzers.chest.bench_press.front_bench_press_analyzer import FrontBenchPressAnalyzer
except Exception:
    FrontBenchPressAnalyzer = None

try:
    from backend.analyzers.chest.bench_press.side_bench_press_analyzer import SideBenchPressAnalyzer
except Exception:
    SideBenchPressAnalyzer = None

try:
    from backend.analyzers.chest.pushup_analyzer import PushupAnalyzer
except Exception:
    PushupAnalyzer = None


def _buildRegistry() -> Dict[str, Type]:
    registry: Dict[str, Type] = {}

    if SquatAnalyzer is not None:
        registry["squat"] = SquatAnalyzer

    if LungeAnalyzer is not None:
        registry["lunge"] = LungeAnalyzer

    if FrontBenchPressAnalyzer is not None:
        registry["bench_front"] = FrontBenchPressAnalyzer

    if SideBenchPressAnalyzer is not None:
        registry["bench_side"] = SideBenchPressAnalyzer

    if PushupAnalyzer is not None:
        registry["pushup"] = PushupAnalyzer

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