import json
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402
from backend.analyzers.back.pull_up_analyzer import PullUpAnalyzer  # noqa: E402


def main():
    defaultVideo = projectRoot / "backend" / "pullup" / "pullup.mp4"

    if len(sys.argv) < 2:
        videoPath = defaultVideo
        outputPathArg = None
        print(f"No video path provided, using default: {videoPath}")
    else:
        videoPath = Path(sys.argv[1])
        if not videoPath.is_absolute():
            videoPath = projectRoot / videoPath
        outputPathArg = sys.argv[2] if len(sys.argv) >= 3 else None

    if not videoPath.exists():
        print(f"ERROR: video file not found: {videoPath}")
        print("Usage: python backend/scripts/test_pull_up_analyzer.py <video_path> [output_json_path]")
        return

    if outputPathArg is None:
        outputDir = projectRoot / "backend" / "results"
        outputDir.mkdir(parents=True, exist_ok=True)
        outputPath = outputDir / "pullup_result.json"
    else:
        outputPath = Path(outputPathArg)
        if not outputPath.is_absolute():
            outputPath = projectRoot / outputPath
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running pull-up back analyzer on: {videoPath}")

    extractionResult = extract_pose_frames(str(videoPath))
    videoMetadata = extractionResult["videoMetadata"]
    poseFrames = extractionResult["poseFrames"]

    analyzer = PullUpAnalyzer()
    result = analyzer.analyze(
        videoPath=str(videoPath),
        poseFrames=poseFrames,
        videoMetadata=videoMetadata,
        options={
            "topElbowAngleDeg": 95,
            "topArmpitAngleDeg": 105,
            "leaveTopElbowDeg": 120,
            "hysteresisDeg": 6,
            "smoothWindow": 5,
            "requireArmpitForHeight": False,
            "enableAngleSymmetry": True,
            "symElbowDiffWarnDeg": 18,
            "symArmpitDiffWarnDeg": 18,
        },
    )

    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Pull-Up Back Analyzer Summary ===")
    print(f"Saved JSON result to: {outputPath}")
    print(f"Status: {result.get('status')}")
    print(f"Rep count: {result.get('repCount')}")
    print(f"Summary score: {result.get('summaryScore')}")
    print(f"Issues: {[i.get('code') for i in result.get('issues', [])]}")
    print(f"Warnings: {result.get('warnings', [])}")


if __name__ == "__main__":
    main()