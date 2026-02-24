import sys
from pathlib import Path


# Allow running the script directly: python backend/scripts/test_pose_extraction.py ...
projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))


from backend.core.pose.mediapipe_extractor import extract_pose_frames  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python backend/scripts/test_pose_extraction.py <video_path>")
        return

    videoPath = sys.argv[1]

    print(f"Processing video: {videoPath}")
    result = extract_pose_frames(videoPath)

    videoMetadata = result["videoMetadata"]
    poseFrames = result["poseFrames"]

    print("\n=== Video Metadata ===")
    print(videoMetadata)

    totalFrames = len(poseFrames)
    framesWithPose = sum(1 for frame in poseFrames if frame.hasPose)

    print("\n=== Pose Extraction Summary ===")
    print(f"Total frames read: {totalFrames}")
    print(f"Frames with pose detected: {framesWithPose}")

    if totalFrames > 0:
        detectionRate = (framesWithPose / totalFrames) * 100
        print(f"Pose detection rate: {detectionRate:.2f}%")

    print("\n=== First 3 Frames (preview) ===")
    for frame in poseFrames[:3]:
        print(frame.to_dict())

    # Optional sample of a specific landmark (if available)
    print("\n=== Sample landmark check (left_knee in first detected frame) ===")
    firstDetectedFrame = next((frame for frame in poseFrames if frame.hasPose), None)
    if firstDetectedFrame is None:
        print("No pose detected in any frame.")
    else:
        leftKnee = firstDetectedFrame.landmarks.get("left_knee")
        if leftKnee is None:
            print("left_knee not found in first detected frame.")
        else:
            print(leftKnee.to_dict())


if __name__ == "__main__":
    main()