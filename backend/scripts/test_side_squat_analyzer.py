import json
import sys
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))


from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402
from backend.analyzers.legs.squat.side_squat_analyzer import SideSquatAnalyzer  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python backend/scripts/test_side_squat_analyzer.py <video_path> "
            "[left|right] [output_json_path]"
        )
        return

    videoPath = sys.argv[1]
    side = sys.argv[2] if len(sys.argv) >= 3 else "left"
    outputPathArg = sys.argv[3] if len(sys.argv) >= 4 else None

    if outputPathArg is None:
        outputDir = projectRoot / "backend" / "results"
        outputDir.mkdir(parents=True, exist_ok=True)
        outputPath = outputDir / "side_squat_result.json"
    else:
        outputPath = Path(outputPathArg)
        if not outputPath.is_absolute():
            outputPath = projectRoot / outputPath
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running side squat analyzer on: {videoPath}")
    print(f"Leg side preference: {side}")

    extractionResult = extract_pose_frames(videoPath)
    videoMetadata = extractionResult["videoMetadata"]
    poseFrames = extractionResult["poseFrames"]

    analyzer = SideSquatAnalyzer()
    result = analyzer.analyze(
        videoPath=videoPath,
        poseFrames=poseFrames,
        videoMetadata=videoMetadata,
        options={"side": side},
    )

    with open(outputPath, "w", encoding="utf-8") as jsonFile:
        json.dump(result, jsonFile, indent=2)

    print("\n=== Side Squat Analyzer Summary ===")
    print(f"Saved JSON result to: {outputPath}")
    print(f"Status: {result.get('status')}")
    print(f"Rep count: {result.get('repCount')}")
    print(f"Summary score: {result.get('summaryScore')}")
    print(f"Top-level issues: {[issue.get('code') for issue in result.get('issues', [])]}")
    print(f"Warnings: {result.get('warnings', [])}")


if __name__ == "__main__":
    main()