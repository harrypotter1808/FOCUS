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

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

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
            image_points = np.array([
                (face_landmarks[i].x * w, face_landmarks[i].y * h)
                for i in LANDMARK_IDX
            ], dtype=np.float64)

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, _ = cv2.solvePnP(
                MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                euler_angles = euler_angles.flatten()
                pitch = float(euler_angles[0])
                yaw = float(euler_angles[1])
                roll = float(euler_angles[2])

                cv2.putText(frame, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                status = "LOOKING AWAY" if abs(yaw) > 25 else "FACING FORWARD"
                color = (0, 0, 255) if status == "LOOKING AWAY" else (0, 255, 0)
                cv2.putText(frame, status, (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Head Pose Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()