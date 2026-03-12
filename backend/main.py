import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.analyzers.registry import getAnalyzer, getSupportedExercises
from backend.core.pose.mediapipe_extractor import extract_pose_frames
from backend.core.video.bench_feedback_video import create_bench_feedback_video
from backend.core.video.front_squat_feedback_video import create_front_squat_feedback_video
from backend.core.video.squat_feedback_video import create_squat_feedback_video
from backend.core.video.standard_feedback_video import create_standard_feedback_video
from backend.feedback.build_text_feedback import build_text_feedback


BACKEND_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Exercise Analyzer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/media/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _safe_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    return extension if extension in ALLOWED_VIDEO_EXTENSIONS else ".mp4"


def _has_video_magic_bytes(filePath: Path) -> bool:
    """
    Lightweight magic-bytes validation for common containers:
    - MP4/MOV/M4V: ISO BMFF with 'ftyp' box
    - AVI: RIFF....AVI
    - MKV/WEBM: EBML header 0x1A45DFA3
    """
    try:
        with filePath.open("rb") as file:
            header = file.read(64)
    except OSError:
        return False

    if len(header) < 12:
        return False

    # MP4/MOV/M4V (ISO BMFF): bytes 4..8 should be 'ftyp'
    if header[4:8] == b"ftyp":
        return True

    # AVI: RIFF + AVI in header
    if header.startswith(b"RIFF") and b"AVI " in header[:16]:
        return True

    # MKV / WEBM: EBML header
    if header.startswith(b"\x1A\x45\xDF\xA3"):
        return True

    return False


def _can_open_as_video(filePath: Path) -> bool:
    capture = cv2.VideoCapture(str(filePath))
    try:
        if not capture.isOpened():
            return False
        frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return frameCount > 0 and fps > 0
    finally:
        capture.release()


def _exercise_default_options(exercise: str, side: Optional[str]) -> Dict[str, Any]:
    sideValue = (side or "").lower().strip()

    if exercise == "squat_side":
        return {"side": sideValue if sideValue in {"left", "right"} else "left"}

    if exercise == "bench_side":
        return {
            "side": sideValue if sideValue in {"left", "right"} else "left",
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
        }

    if exercise == "bench_front":
        return {
            "topAngleDeg": 155,
            "bottomAngleDeg": 95,
            "holdFrames": 4,
            "minRomDeg": 60,
            "bottomMarginDeg": 5,
            "topMarginDeg": 5,
            "gripMinRatio": 1.0,
            "gripMaxRatio": 1.5,
            "wristYDiffWarn": 0.04,
            "elbowAngleDiffWarn": 15.0,
            "midXDriftWarn": 0.06,
        }

    if exercise == "pullup_back":
        return {
            "topElbowAngleDeg": 95,
            "topArmpitAngleDeg": 105,
            "leaveTopElbowDeg": 120,
            "hysteresisDeg": 6,
            "smoothWindow": 5,
            "requireArmpitForHeight": False,
            "enableAngleSymmetry": True,
            "symElbowDiffWarnDeg": 18,
            "symArmpitDiffWarnDeg": 18,
        }

    if exercise == "bicep_curl_side":
        return {
            "side": sideValue if sideValue in {"left", "right"} else "right",
            "bottomAngleDeg": 158,
            "topAngleDeg": 68,
            "hysteresisDeg": 6,
            "smoothWindow": 5,
            "minRomDeg": 70,
            "bottomMarginDeg": 12,
            "topMarginDeg": 8,
            "elbowRelXDriftWarn": 0.27,
        }

    if exercise == "tricep_extension_side":
        return {
            "side": sideValue if sideValue in {"left", "right"} else "left",
            "topAngleDeg": 160,
            "bottomAngleDeg": 85,
            "hysteresisDeg": 5,
            "smoothWindow": 5,
            "minRomDeg": 65,
            "bottomMarginDeg": 5,
            "topMarginDeg": 5,
            "elbowRelXDriftWarn": 0.05,
            "upperArmAngleDriftWarn": 20.0,
        }

    return {}


def _create_feedback_video_for_exercise(
    exercise: str,
    videoPath: str,
    poseFrames: list,
    analysisResult: Dict[str, Any],
    outputPath: str,
) -> Optional[str]:
    if exercise == "squat_side":
        return create_squat_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            pauseSeconds=4.0,
        )

    if exercise == "squat_front":
        return create_front_squat_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            pauseSeconds=4.0,
        )

    if exercise == "bench_side":
        return create_bench_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            panelTitle="Bench Press (Side)",
            pauseSeconds=2.0,
        )

    if exercise == "bench_front":
        return create_bench_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            panelTitle="Bench Press (Front)",
            pauseSeconds=2.0,
        )

    if exercise == "pullup_back":
        return create_standard_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            panelTitle="Pull-Up",
            issueMessages={
                "height_incomplete": "Pull higher before ending the rep.",
                "pull_asymmetry": "Keep the pull more even on both sides.",
            },
            positiveDetailLines=["Pull height and left/right balance looked good."],
            pauseSeconds=4.0,
        )

    if exercise == "bicep_curl_side":
        return create_standard_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            panelTitle="Bicep Curl",
            issueMessages={
                "rom_incomplete": "Use a fuller curl range of motion.",
                "elbow_drift": "Keep your elbow steadier near your torso.",
            },
            positiveDetailLines=["Range of motion and elbow control looked good."],
            pauseSeconds=4.0,
        )

    if exercise == "tricep_extension_side":
        return create_standard_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            panelTitle="Tricep Extension",
            issueMessages={
                "rom_incomplete": "Use a fuller range of motion.",
                "elbow_drift": "Keep your elbow in a steadier position.",
                "upper_arm_instability": "Keep your upper arm more stable.",
            },
            positiveDetailLines=["Range of motion and arm position looked good."],
            pauseSeconds=4.0,
        )

    if exercise == "shoulder_press_front":
        return create_standard_feedback_video(
            videoPath=videoPath,
            poseFrames=poseFrames,
            analysisResult=analysisResult,
            outputPath=outputPath,
            panelTitle="Shoulder Press",
            issueMessages={
                "bottom_bend_shallow": "Lower the weight more before pressing.",
                "lockout_incomplete": "Finish with a stronger lockout overhead.",
                "top_reach_low": "Press higher at the top.",
                "top_symmetry_high_imbalance": "Keep both arms more even at the top.",
                "top_symmetry_mild_imbalance": "Try to level both arms out at the top.",
            },
            positiveDetailLines=["Press depth, lockout, and symmetry looked good."],
            pauseSeconds=4.0,
        )

    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/exercises")
def list_exercises():
    return {"exercises": getSupportedExercises()}


@app.post("/api/analyze")
async def analyze_video(
    exercise: str = Form(...),
    video: UploadFile = File(...),
    side: Optional[str] = Form(None),
):
    supported = set(getSupportedExercises())
    if exercise not in supported:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported exercise '{exercise}'.",
                "supportedExercises": sorted(supported),
            },
        )

    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing video filename.")

    runId = uuid.uuid4().hex[:12]
    extension = _safe_extension(video.filename)
    uploadedPath = UPLOADS_DIR / f"{runId}_{exercise}{extension}"

    try:
        with uploadedPath.open("wb") as outputFile:
            shutil.copyfileobj(video.file, outputFile)
    finally:
        await video.close()

    if not _has_video_magic_bytes(uploadedPath):
        uploadedPath.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a supported video format (magic-bytes check failed).",
        )

    if not _can_open_as_video(uploadedPath):
        uploadedPath.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as a valid video.",
        )

    try:
        extractionResult = extract_pose_frames(str(uploadedPath))
        videoMetadata = extractionResult["videoMetadata"]
        poseFrames = extractionResult["poseFrames"]

        analyzer = getAnalyzer(exercise)
        options = _exercise_default_options(exercise, side)
        result = analyzer.analyze(
            videoPath=str(uploadedPath),
            poseFrames=poseFrames,
            videoMetadata=videoMetadata,
            options=options,
        )

        resultJsonPath = RESULTS_DIR / f"{runId}_{exercise}_result.json"
        with resultJsonPath.open("w", encoding="utf-8") as resultFile:
            json.dump(result, resultFile, indent=2)

        feedbackVideoPath = RESULTS_DIR / f"{runId}_{exercise}_feedback.mp4"
        feedbackVideo = _create_feedback_video_for_exercise(
            exercise=exercise,
            videoPath=str(uploadedPath),
            poseFrames=poseFrames,
            analysisResult=result,
            outputPath=str(feedbackVideoPath),
        )
        textFeedback = build_text_feedback(exercise, result)

        response = {
            "runId": runId,
            "exercise": exercise,
            "analysis": result,
            "textFeedback": textFeedback,
            "uploadedVideoUrl": f"/media/uploads/{uploadedPath.name}",
            "resultJsonUrl": f"/media/results/{resultJsonPath.name}",
            "feedbackVideoUrl": (
                f"/media/results/{Path(feedbackVideo).name}" if feedbackVideo else None
            ),
        }
        return response
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}") from error
