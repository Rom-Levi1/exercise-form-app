import json
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402
from backend.analyzers.chest.bench_press.side_bench_press_analyzer import (  # noqa: E402
    SideBenchPressAnalyzer,
)
from backend.core.video.bench_feedback_video import create_bench_feedback_video  # noqa: E402


def main():
    # Default video path (relative to project root)
    defaultVideo = projectRoot / "backend" / "barbell-bench-press" / "bench_side.mp4"

    # --- Parse args (with defaults) ---
    if len(sys.argv) < 2:
        videoPath = str(defaultVideo)
        side = "left"
        outputPathArg = None
        outputVideoPathArg = None
        print(f"No video path provided, using default: {videoPath}")
    else:
        videoPath = sys.argv[1]
        side = sys.argv[2].lower() if len(sys.argv) >= 3 else "left"
        outputPathArg = sys.argv[3] if len(sys.argv) >= 4 else None
        outputVideoPathArg = sys.argv[4] if len(sys.argv) >= 5 else None

    # Make relative paths relative to project root
    videoPathP = Path(videoPath)
    if not videoPathP.is_absolute():
        videoPathP = projectRoot / videoPathP

    if not videoPathP.exists():
        print(f"ERROR: video file not found: {videoPathP}")
        print(
            "Usage: python backend/scripts/test_side_bench_analyzer.py <video_path> "
            "[left|right] [output_json_path] [output_video_path]"
        )
        return

    # --- Output path ---
    if outputPathArg is None:
        outputDir = projectRoot / "backend" / "results"
        outputDir.mkdir(parents=True, exist_ok=True)
        outputPath = outputDir / f"side_bench_result_{side}.json"
    else:
        outputPath = Path(outputPathArg)
        if not outputPath.is_absolute():
            outputPath = projectRoot / outputPath
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    if outputVideoPathArg is None:
        outputVideoPath = outputPath.with_name(f"side_bench_feedback_{side}.mp4")
    else:
        outputVideoPath = Path(outputVideoPathArg)
        if not outputVideoPath.is_absolute():
            outputVideoPath = projectRoot / outputVideoPath
        outputVideoPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running side bench analyzer on: {videoPathP} (side={side})")

    extractionResult = extract_pose_frames(str(videoPathP))
    videoMetadata = extractionResult["videoMetadata"]
    poseFrames = extractionResult["poseFrames"]

    analyzer = SideBenchPressAnalyzer()
    result = analyzer.analyze(
        videoPath=str(videoPathP),
        poseFrames=poseFrames,
        videoMetadata=videoMetadata,
        options={
            "side": side,
            "topAngleDeg": 155,
            "bottomAngleDeg": 105,
            "hysteresisDeg": 5,
            "smoothWindow": 5,

            "minRomDeg": 50,
            "bottomMarginDeg": 5,
            "topMarginDeg": 5,

            "wristElbowXWarn": 0.04,
            "barRelXDriftWarn": 0.06,

            "enableTuckCheck": False,
        },
    )

    with open(outputPath, "w", encoding="utf-8") as jsonFile:
        json.dump(result, jsonFile, indent=2)

    annotatedVideoPath = create_bench_feedback_video(
        videoPath=str(videoPathP),
        poseFrames=poseFrames,
        analysisResult=result,
        outputPath=str(outputVideoPath),
        panelTitle="Bench Press",
        pauseSeconds=4.0,
    )

    print("\n=== Side Bench Analyzer Summary ===")
    print(f"Saved JSON result to: {outputPath}")
    print(f"Status: {result.get('status')}")
    print(f"Rep count: {result.get('repCount')}")
    print(f"Summary score: {result.get('summaryScore')}")
    print(f"Top-level issues: {[issue.get('code') for issue in result.get('issues', [])]}")
    print(f"Warnings: {result.get('warnings', [])}")
    print(f"Annotated video: {annotatedVideoPath or 'Failed to generate'}")

    reps = result.get("repFeedback", []) or []
    if reps:
        print("\nFirst repFeedback entries:")
        for r in reps[:3]:
            print(
                f"rep={r.get('repIndex')} romOk={r.get('romOk')} romDeg={r.get('romDeg')} "
                f"barPathOk={r.get('barPathOk')} drift={r.get('barRelXDrift')} "
                f"stackingOk={r.get('stackingOk')} badStackRatio={r.get('stackingBadRatio')}"
            )


if __name__ == "__main__":
    main()
