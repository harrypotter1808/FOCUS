import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=5,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.FaceLandmarker.create_from_options(options)

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
LEFT_EYEBROW = 105
LEFT_EYE_TOP = 159

def get_point(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 33

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            mouth_top = get_point(face_landmarks, MOUTH_TOP, w, h)
            mouth_bottom = get_point(face_landmarks, MOUTH_BOTTOM, w, h)
            mouth_left = get_point(face_landmarks, MOUTH_LEFT, w, h)
            mouth_right = get_point(face_landmarks, MOUTH_RIGHT, w, h)
            eyebrow = get_point(face_landmarks, LEFT_EYEBROW, w, h)
            eye_top = get_point(face_landmarks, LEFT_EYE_TOP, w, h)

            mouth_gap = np.linalg.norm(mouth_top - mouth_bottom)
            mouth_width = np.linalg.norm(mouth_left - mouth_right)
            mouth_ratio = mouth_gap / mouth_width if mouth_width != 0 else 0

            eyebrow_gap = np.linalg.norm(eyebrow - eye_top)

            score = 0.7
            if mouth_ratio > 0.5:
                score -= 0.3
            if eyebrow_gap < 8:
                score -= 0.2
            score = max(0.0, min(1.0, score))

            cv2.putText(frame, f"Expression score: {score:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouth ratio: {mouth_ratio:.2f}  Eyebrow gap: {eyebrow_gap:.1f}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            yawn_status = "YAWN DETECTED" if mouth_ratio > 0.5 else ""
            cv2.putText(frame, yawn_status, (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            brow_status = "LOW EYEBROWS (blank/drowsy look)" if eyebrow_gap < 8 else ""
            cv2.putText(frame, brow_status, (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Expression Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()