import json
import sys
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))


from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402
from backend.analyzers.legs.squat.front_squat_analyzer import FrontSquatAnalyzer  # noqa: E402
from backend.core.video.front_squat_feedback_video import create_front_squat_feedback_video  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python backend/scripts/test_front_squat_analyzer.py <video_path> "
            "[output_json_path] [output_video_path]"
        )
        return

    videoPath = sys.argv[1]
    outputPathArg = sys.argv[2] if len(sys.argv) >= 3 else None
    outputVideoPathArg = sys.argv[3] if len(sys.argv) >= 4 else None

    if outputPathArg is None:
        outputDir = projectRoot / "backend" / "results"
        outputDir.mkdir(parents=True, exist_ok=True)
        outputPath = outputDir / "front_squat_result.json"
    else:
        outputPath = Path(outputPathArg)
        if not outputPath.is_absolute():
            outputPath = projectRoot / outputPath
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    if outputVideoPathArg is None:
        outputVideoPath = outputPath.with_name("front_squat_feedback.mp4")
    else:
        outputVideoPath = Path(outputVideoPathArg)
        if not outputVideoPath.is_absolute():
            outputVideoPath = projectRoot / outputVideoPath
        outputVideoPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running front squat analyzer on: {videoPath}")

    extractionResult = extract_pose_frames(videoPath)
    videoMetadata = extractionResult["videoMetadata"]
    poseFrames = extractionResult["poseFrames"]

    analyzer = FrontSquatAnalyzer()
    result = analyzer.analyze(
        videoPath=videoPath,
        poseFrames=poseFrames,
        videoMetadata=videoMetadata,
        options={},
    )

    with open(outputPath, "w", encoding="utf-8") as jsonFile:
        json.dump(result, jsonFile, indent=2)

    annotatedVideoPath = create_front_squat_feedback_video(
        videoPath=videoPath,
        poseFrames=poseFrames,
        analysisResult=result,
        outputPath=str(outputVideoPath),
        pauseSeconds=4.0,
    )

    print("\n=== Front Squat Analyzer Summary ===")
    print(f"Saved JSON result to: {outputPath}")
    print(f"Status: {result.get('status')}")
    print(f"Rep count: {result.get('repCount')} (expected: 0 for front-view clip-level mode)")
    print(f"Summary score: {result.get('summaryScore')}")
    print(f"Top-level issues: {[issue.get('code') for issue in result.get('issues', [])]}")
    print(f"Warnings: {result.get('warnings', [])}")
    print(f"Annotated video: {annotatedVideoPath or 'Failed to generate'}")

    metrics = result.get("metrics", {})
    repFeedback = result.get("repFeedback", [])

    print("\n=== Front Squat Clip-Level Checks ===")
    print(f"Analysis mode: {metrics.get('analysisMode')}")
    print(f"View: {metrics.get('view')}")

    if repFeedback:
        clipSummary = repFeedback[0]
        checks = clipSummary.get("checks", {})

        stanceWidth = checks.get("stanceWidth", {})
        symmetry = checks.get("symmetry", {})

        print(f"Clip quality: {clipSummary.get('quality')}")
        print(f"Clip issues: {clipSummary.get('issues', [])}")

        print("\nStance Width:")
        print(f"  Classification: {stanceWidth.get('classification')}")
        print(f"  Avg stance/shoulder ratio: {stanceWidth.get('avgStanceToShoulderRatio')}")
        print(f"  Min stance/shoulder ratio: {stanceWidth.get('minStanceToShoulderRatio')}")
        print(f"  Max stance/shoulder ratio: {stanceWidth.get('maxStanceToShoulderRatio')}")

        print("\nSymmetry:")
        print(f"  Classification: {symmetry.get('classification')}")
        print(f"  Avg normalized imbalance: {symmetry.get('avgNormalizedImbalance')}")
        print(f"  Min normalized imbalance: {symmetry.get('minNormalizedImbalance')}")
        print(f"  Max normalized imbalance: {symmetry.get('maxNormalizedImbalance')}")
    else:
        print("No clip-level feedback returned.")

    print("\n=== Front Squat Validation Checklist ===")
    print(f"status == 'success': {result.get('status') == 'success'}")
    print(f"repCount == 0: {result.get('repCount') == 0}")
    print(f"summaryScore is not None: {result.get('summaryScore') is not None}")
    print(f"analysisMode == 'clip_level_only': {metrics.get('analysisMode') == 'clip_level_only'}")
    print(f"repFeedback has one entry: {len(repFeedback) == 1}")


if __name__ == "__main__":
    main()
