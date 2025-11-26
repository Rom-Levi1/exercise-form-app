import cv2
import numpy as np
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates angle (in degrees) at point b given three 2D points a, b, c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def main():
    video_path = "squat.mp4"   # 🟢 change to your file name
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: could not open video:", video_path)
        return
    
    paused = False  # play/pause state

    window_name = "Squat Video Analysis"
    cv2.namedWindow(window_name)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def on_trackbar(val):
        # jump to selected frame when user drags the bar
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    cv2.createTrackbar("Position", window_name, 0, max(frame_count - 1, 1), on_trackbar)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video   

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            feedback_text = "No pose detected"

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                landmarks = results.pose_landmarks.landmark

                try:
                    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                    hip_xy = (hip.x, hip.y)
                    knee_xy = (knee.x, knee.y)
                    ankle_xy = (ankle.x, ankle.y)

                    angle = calculate_angle(hip_xy, knee_xy, ankle_xy)
                    feedback_text = f"Knee angle: {int(angle)}°"
                    # Squat depth
                    if angle < 80:
                        feedback_text = "Good depth"
                    elif angle < 140:
                        feedback_text = "Go lower"
                    else:
                        feedback_text = "Starting position"

                    # 🟠 Knee over toes check (X axis)
                    left_toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                    knee_toe_diff = left_toe.x - knee.x  # normalized [-1,1] space in ratio

                    if knee_toe_diff > 0.2:  # knee passed toes too much
                        feedback_text += " | Knee too far over toes"
                    elif knee_toe_diff > 0.02:
                        feedback_text += " | Knee slightly over toes"
                    else:
                        feedback_text += " | Knee alignment good"


                    h, w, _ = image_bgr.shape
                    knee_px = (int(knee.x * w), int(knee.y * h))

                    cv2.putText(
                        image_bgr,
                        str(int(angle)) + " deg",
                        (knee_px[0] + 10, knee_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                except:
                    feedback_text = "Move into frame"

            # Feedback text at the top
            cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                image_bgr,
                feedback_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # show frame
            cv2.imshow(window_name, image_bgr)

            # update slider position to current frame
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Position", window_name, current_frame)

            # wait for key: if paused, wait forever until a key; else ~20ms
            key = cv2.waitKey(0 if paused else 20) & 0xFF

            if key == ord('q' | 'Q'):
                break  # quit

            elif key == 32:  # SPACE = play/pause
                paused = not paused

            elif key == ord('s'):  # 's' = stop (pause and go to start)
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            elif key == ord('d'):  # 'd' = step forward 1 frame
                paused = True  # stay paused
                # nothing else; next loop iteration will read next frame

            elif key == ord('a'):  # 'a' = step back ~1 frame
                paused = True
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
