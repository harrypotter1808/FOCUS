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

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308
LEFT_EYEBROW, LEFT_EYE_TOP = 105, 159
LANDMARK_IDX = [1, 152, 33, 263, 61, 291]
BASELINE_YAW = -15.0  # use YOUR measured value from test_headpose.py

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)

WEIGHTS = {"expression": 0.3, "head_pose": 0.3, "drowsiness": 0.4}
ALERT_THRESHOLD = 0.45

def get_point(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (vert1 + vert2) / (2.0 * horiz) if horiz != 0 else 0.3

def drowsiness_from_ear(ear, closed=0.19, opened=0.30):
    if ear <= closed:
        return 1.0
    if ear >= opened:
        return 0.0
    return 1.0 - (ear - closed) / (opened - closed)

def head_pose_deviation(landmarks, w, h):
    image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDX], dtype=np.float64)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, _ = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.5
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    euler_angles = euler_angles.flatten()
    yaw = float(euler_angles[1])
    deviation = abs(yaw - BASELINE_YAW)
    return min(deviation / 45.0, 1.0)

def expression_score(landmarks, w, h):
    mouth_top = get_point(landmarks, MOUTH_TOP, w, h)
    mouth_bottom = get_point(landmarks, MOUTH_BOTTOM, w, h)
    mouth_left = get_point(landmarks, MOUTH_LEFT, w, h)
    mouth_right = get_point(landmarks, MOUTH_RIGHT, w, h)
    eyebrow = get_point(landmarks, LEFT_EYEBROW, w, h)
    eye_top = get_point(landmarks, LEFT_EYE_TOP, w, h)

    mouth_gap = np.linalg.norm(mouth_top - mouth_bottom)
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    mouth_ratio = mouth_gap / mouth_width if mouth_width != 0 else 0
    eyebrow_gap = np.linalg.norm(eyebrow - eye_top)

    score = 0.7
    if mouth_ratio > 0.5:
        score -= 0.3
    if eyebrow_gap < 8:
        score -= 0.2
    return max(0.0, min(1.0, score))

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
        for landmarks in result.face_landmarks:
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            drowsy = drowsiness_from_ear(ear)
            pose_dev = head_pose_deviation(landmarks, w, h)
            expr = expression_score(landmarks, w, h)

            engagement = (
                WEIGHTS["expression"] * expr
                + WEIGHTS["head_pose"] * (1 - pose_dev)
                + WEIGHTS["drowsiness"] * (1 - drowsy)
            )
            engagement = round(max(0.0, min(1.0, engagement)), 2)

            status = "DISENGAGED" if engagement < ALERT_THRESHOLD else "ENGAGED"
            color = (0, 0, 255) if status == "DISENGAGED" else (0, 255, 0)

            cv2.putText(frame, f"Engagement: {engagement:.2f} [{status}]", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"expr={expr:.2f} pose_dev={pose_dev:.2f} drowsy={drowsy:.2f}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Engagement Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()