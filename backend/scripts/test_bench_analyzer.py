import sys
from pathlib import Path

# Allow running directly from anywhere
projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from backend.core.pose.mediapipe_extractor import extract_pose_frames
from backend.analyzers.registry import getAnalyzer


def print_report(result: dict):
    print("\n==============================")
    print(f"Exercise:     {result.get('exercise')}")
    print(f"Status:       {result.get('status')}")
    print(f"Message:      {result.get('message')}")
    print("------------------------------")

    print(f"Rep Count:    {result.get('repCount')}")
    print(f"SummaryScore: {result.get('summaryScore')}")
    print("------------------------------")

    metrics = result.get("metrics", {})
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
    print("------------------------------")

    issues = result.get("issues", [])
    print(f"Issues ({len(issues)}):")
    if not issues:
        print("  - None")
    else:
        for i, it in enumerate(issues, 1):
            print(f"  {i}) [{it.get('severity')}] {it.get('code')}: {it.get('message')}")
    print("------------------------------")

    reps = result.get("repFeedback", [])
    print(f"Per-Rep Feedback ({len(reps)}):")
    if not reps:
        print("  - None")
    else:
        for r in reps:
            print(
                f"  Rep {r.get('repIndex')}: "
                f"depthOk={r.get('depthOk')} | "
                f"flareOk={r.get('flareOk')} | "
                f"flareBadRatio={r.get('flareBadRatio')}"
            )

    warnings = result.get("warnings", [])
    print("------------------------------")
    print(f"Warnings ({len(warnings)}):")
    if not warnings:
        print("  - None")
    else:
        for w in warnings:
            print(f"  - {w}")

    print("==============================\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python backend/scripts/test_bench_analyzer.py bench_press <video_path>")
        return

    exercise = sys.argv[1]
    videoPath = sys.argv[2]

    extracted = extract_pose_frames(videoPath)
    analyzer = getAnalyzer(exercise)

    result = analyzer.analyze(
        videoPath=videoPath,
        poseFrames=extracted["poseFrames"],
        videoMetadata=extracted["videoMetadata"],
        options={
            # tweak these if needed
            "topAngleDeg": 155,
            "bottomAngleDeg": 95,
            "holdFrames": 4,
            "flareMinDeg": 45,
            "flareMaxDeg": 75,
        }
    )

    print_report(result)


if __name__ == "__main__":
    main()