import cv2
import numpy as np
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at point b,
    where a, b, c are 2D points: (x, y).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # cosine formula
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def main():
    # for webcam
    # cap = cv2.VideoCapture(0)

    # if you want a video instead:
    cap = cv2.VideoCapture("bench_front.mp4")

    if not cap.isOpened():
        print("Error: could not open camera or video")
        return

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No more frames or camera error")
                break

            # optional: flip so it looks like a mirror
            frame = cv2.flip(frame, 1)

            # convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            # back to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            h, w, _ = image_bgr.shape

            feedback_text = "No person detected"
            angle_text = ""

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                landmarks = results.pose_landmarks.landmark

                try:
                    # use right arm for example
                    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                    # normalized coords for angle calculation
                    shoulder_xy = (shoulder.x, shoulder.y)
                    elbow_xy = (elbow.x, elbow.y)
                    wrist_xy = (wrist.x, wrist.y)

                    elbow_angle = calculate_angle(shoulder_xy, elbow_xy, wrist_xy)
                    angle_text = f"Elbow angle: {int(elbow_angle)}"

                    # convert elbow position to pixels for drawing text
                    elbow_px = (int(elbow.x * w), int(elbow.y * h))

                    # simple logic for bench depth
                    # you will probably tweak these values after testing
                    if elbow_angle < 80:
                        feedback_text = "Good depth"
                    elif elbow_angle < 120:
                        feedback_text = "Lower the bar more"
                    else:
                        feedback_text = "Arms almost straight"

                    # draw the angle text near the elbow
                    cv2.putText(
                        image_bgr,
                        angle_text,
                        (elbow_px[0] + 10, elbow_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                except Exception as e:
                    feedback_text = "Cannot see full arm"

            # top bar with feedback
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

            cv2.imshow("Bench Press Elbow Analysis (press q to quit)", image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
