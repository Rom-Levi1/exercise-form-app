import json
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402
from backend.analyzers.arms.tricep_extension.side_tricep_extension_analyzer import (  # noqa: E402
    SideTricepExtensionAnalyzer,
)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python backend/scripts/test_side_tricep_extension_analyzer.py <video_path> "
            "[left|right] [output_json_path]"
        )
        return

    videoPath = sys.argv[1]
    sideArg = sys.argv[2] if len(sys.argv) >= 3 else "left"
    outputPathArg = sys.argv[3] if len(sys.argv) >= 4 else None

    if outputPathArg is None:
        outputDir = projectRoot / "backend" / "results"
        outputDir.mkdir(parents=True, exist_ok=True)
        outputPath = outputDir / "side_tricep_extension_result.json"
    else:
        outputPath = Path(outputPathArg)
        if not outputPath.is_absolute():
            outputPath = projectRoot / outputPath
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running side tricep extension analyzer on: {videoPath} (side={sideArg})")

    extractionResult = extract_pose_frames(videoPath)
    videoMetadata = extractionResult["videoMetadata"]
    poseFrames = extractionResult["poseFrames"]

    analyzer = SideTricepExtensionAnalyzer()
    result = analyzer.analyze(
        videoPath=videoPath,
        poseFrames=poseFrames,
        videoMetadata=videoMetadata,
        options={
            "side": sideArg,
            "topAngleDeg": 160,
            "bottomAngleDeg": 85,
            "hysteresisDeg": 5,
            "smoothWindow": 5,
            "minRomDeg": 65,
            "bottomMarginDeg": 5,
            "topMarginDeg": 5,
            "elbowRelXDriftWarn": 0.05,
            "upperArmAngleDriftWarn": 20.0,
        },
    )

    with open(outputPath, "w", encoding="utf-8") as jsonFile:
        json.dump(result, jsonFile, indent=2)

    print("\n=== Side Tricep Extension Analyzer Summary ===")
    print(f"Saved JSON result to: {outputPath}")
    print(f"Status: {result.get('status')}")
    print(f"Rep count: {result.get('repCount')}")
    print(f"Summary score: {result.get('summaryScore')}")
    print(f"Top-level issues: {[issue.get('code') for issue in result.get('issues', [])]}")
    print(f"Warnings: {result.get('warnings', [])}")


if __name__ == "__main__":
    main()
