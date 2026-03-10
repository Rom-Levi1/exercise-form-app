import json
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402
from backend.analyzers.chest.bench_press.front_bench_press_analyzer import (  # noqa: E402
    FrontBenchPressAnalyzer,
)
from backend.core.video.bench_feedback_video import create_bench_feedback_video  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python backend/scripts/test_front_bench_analyzer.py <video_path> "
            "[output_json_path] [output_video_path]"
        )
        return

    videoPath = sys.argv[1]
    outputPathArg = sys.argv[2] if len(sys.argv) >= 3 else None
    outputVideoPathArg = sys.argv[3] if len(sys.argv) >= 4 else None

    if outputPathArg is None:
        outputDir = projectRoot / "backend" / "results"
        outputDir.mkdir(parents=True, exist_ok=True)
        outputPath = outputDir / "front_bench_result.json"
    else:
        outputPath = Path(outputPathArg)
        if not outputPath.is_absolute():
            outputPath = projectRoot / outputPath
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    if outputVideoPathArg is None:
        outputVideoPath = outputPath.with_name("front_bench_feedback.mp4")
    else:
        outputVideoPath = Path(outputVideoPathArg)
        if not outputVideoPath.is_absolute():
            outputVideoPath = projectRoot / outputVideoPath
        outputVideoPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running front bench analyzer on: {videoPath}")

    extractionResult = extract_pose_frames(videoPath)
    videoMetadata = extractionResult["videoMetadata"]
    poseFrames = extractionResult["poseFrames"]

    analyzer = FrontBenchPressAnalyzer()
    result = analyzer.analyze(
        videoPath=videoPath,
        poseFrames=poseFrames,
        videoMetadata=videoMetadata,
        options={
            "topAngleDeg": 155,
            "bottomAngleDeg": 95,
            "holdFrames": 4,

            # NEW: ROM validation
            "minRomDeg": 60,
            "bottomMarginDeg": 5,
            "topMarginDeg": 5,

            "gripMinRatio": 1.0,
            "gripMaxRatio": 1.5,
            "wristYDiffWarn": 0.04,
            "elbowAngleDiffWarn": 15.0,
            "midXDriftWarn": 0.06,
        },
    )

    with open(outputPath, "w", encoding="utf-8") as jsonFile:
        json.dump(result, jsonFile, indent=2)

    annotatedVideoPath = create_bench_feedback_video(
        videoPath=videoPath,
        poseFrames=poseFrames,
        analysisResult=result,
        outputPath=str(outputVideoPath),
        panelTitle="Bench Press",
        pauseSeconds=4.0,
    )

    print("\n=== Front Bench Analyzer Summary ===")
    print(f"Saved JSON result to: {outputPath}")
    print(f"Status: {result.get('status')}")
    print(f"Rep count: {result.get('repCount')}")
    print(f"Summary score: {result.get('summaryScore')}")
    print(f"Top-level issues: {[issue.get('code') for issue in result.get('issues', [])]}")
    print(f"Warnings: {result.get('warnings', [])}")
    print(f"Annotated video: {annotatedVideoPath or 'Failed to generate'}")


if __name__ == "__main__":
    main()
