import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(projectRoot))

from backend.core.pose.mediapipe_extractor import extract_pose_frames
from backend.analyzers.chest.bench_press.front_bench_press_analyzer import FrontBenchPressAnalyzer


def print_report(result: dict):
    print("\n==============================")
    print(f"Exercise:  {result.get('exercise')}")
    print(f"Status:    {result.get('status')}")
    print(f"Message:   {result.get('message')}")
    print("------------------------------")
    print(f"RepCount:  {result.get('repCount')}")
    print(f"Score:     {result.get('summaryScore')}")
    print("------------------------------")
    print("Issues:")
    for it in (result.get("issues") or []):
        print(f"  - [{it.get('severity')}] {it.get('code')}: {it.get('message')}")
    if not (result.get("issues") or []):
        print("  - None")
    print("------------------------------")
    print("Per rep:")
    for r in (result.get("repFeedback") or []):
        print(f"  Rep {r['repIndex']}: symmetryOk={r['symmetryOk']} (bad={r['symmetryBadRatio']}), centeredOk={r['barCenteredOk']} (drift={r['midXDrift']})")
    print("==============================\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python backend/scripts/test_front_bench.py <video_path>")
        return

    videoPath = sys.argv[1]
    extracted = extract_pose_frames(videoPath)

    analyzer = FrontBenchPressAnalyzer()
    result = analyzer.analyze(
        videoPath=videoPath,
        poseFrames=extracted["poseFrames"],
        videoMetadata=extracted["videoMetadata"],
        options={}
    )
    print_report(result)


if __name__ == "__main__":
    main()