import json
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402
from backend.analyzers.arms.bicep_curl.side_bicep_curl_analyzer import (  # noqa: E402
    SideBicepCurlAnalyzer,
)
from backend.core.video.standard_feedback_video import create_standard_feedback_video  # noqa: E402


def main():
    defaultVideo = projectRoot / "backend" / "bicep-curl" / "bicep_curl.mp4"

    if len(sys.argv) < 2:
        videoPath = defaultVideo
        sideArg = "right"
        outputPathArg = None
        outputVideoPathArg = None
        print(f"No video path provided, using default: {videoPath}")
    else:
        videoPath = Path(sys.argv[1])
        if not videoPath.is_absolute():
            videoPath = projectRoot / videoPath
        sideArg = sys.argv[2] if len(sys.argv) >= 3 else "right"
        outputPathArg = sys.argv[3] if len(sys.argv) >= 4 else None
        outputVideoPathArg = sys.argv[4] if len(sys.argv) >= 5 else None

    if not videoPath.exists():
        print(f"ERROR: video file not found: {videoPath}")
        print(
            "Usage: python backend/scripts/test_side_bicep_curl_analyzer.py <video_path> "
            "[left|right] [output_json_path] [output_video_path]"
        )
        return

    if outputPathArg is None:
        outputDir = projectRoot / "backend" / "results"
        outputDir.mkdir(parents=True, exist_ok=True)
        outputPath = outputDir / "side_bicep_curl_result.json"
    else:
        outputPath = Path(outputPathArg)
        if not outputPath.is_absolute():
            outputPath = projectRoot / outputPath
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    if outputVideoPathArg is None:
        outputVideoPath = outputPath.with_name("side_bicep_curl_feedback.mp4")
    else:
        outputVideoPath = Path(outputVideoPathArg)
        if not outputVideoPath.is_absolute():
            outputVideoPath = projectRoot / outputVideoPath
        outputVideoPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running side bicep curl analyzer on: {videoPath} (side={sideArg})")

    extractionResult = extract_pose_frames(str(videoPath))
    videoMetadata = extractionResult["videoMetadata"]
    poseFrames = extractionResult["poseFrames"]

    analyzer = SideBicepCurlAnalyzer()
    result = analyzer.analyze(
        videoPath=str(videoPath),
        poseFrames=poseFrames,
        videoMetadata=videoMetadata,
        options={
            "side": sideArg,
            "bottomAngleDeg": 158,
            "topAngleDeg": 68,
            "hysteresisDeg": 6,
            "smoothWindow": 5,
            "minRomDeg": 70,
            "bottomMarginDeg": 12,
            "topMarginDeg": 8,
            "minDetectRomDeg": 35,
            "detectionBottomSlackDeg": 12,
            "detectionTopSlackDeg": 12,
            "ascentStartRomDeg": 25,
            "elbowRelXDriftWarn": 0.27,
            "upperArmAngleDriftWarn": 20.0,
        },
    )

    with open(outputPath, "w", encoding="utf-8") as jsonFile:
        json.dump(result, jsonFile, indent=2)

    annotatedVideoPath = create_standard_feedback_video(
        videoPath=str(videoPath),
        poseFrames=poseFrames,
        analysisResult=result,
        outputPath=str(outputVideoPath),
        panelTitle="Bicep Curl",
        issueMessages={
            "rom_incomplete": "Use a fuller curl range of motion.",
            "bottom_position_incomplete": "Lower the weight more before starting the curl.",
            "top_position_incomplete": "Finish the curl higher at the top.",
            "elbow_drift": "Keep your elbow steadier near your torso.",
            "upper_arm_instability": "Keep your upper arm steadier during the curl.",
        },
        positiveDetailLines=["Range of motion and arm path looked good."],
        pauseSeconds=4.0,
    )

    print("\n=== Side Bicep Curl Analyzer Summary ===")
    print(f"Saved JSON result to: {outputPath}")
    print(f"Status: {result.get('status')}")
    print(f"Rep count: {result.get('repCount')}")
    print(f"Summary score: {result.get('summaryScore')}")
    print(f"Top-level issues: {[issue.get('code') for issue in result.get('issues', [])]}")
    print(f"Warnings: {result.get('warnings', [])}")
    print(f"Annotated video: {annotatedVideoPath or 'Failed to generate'}")


if __name__ == "__main__":
    main()
