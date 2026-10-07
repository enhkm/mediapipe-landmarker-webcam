"""gesture_app.py와 gesture_train.py가 함께 쓰는 코드"""
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).parent
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
DATA_PATH = BASE_DIR / "gesture_data.csv"
GESTURE_MODEL_PATH = BASE_DIR / "gesture_model.joblib"
WEB_MODEL_PATH = BASE_DIR / "web" / "gesture_model.json"
CAMERA_INDEX = 0

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 여므로 바이트로 직접 읽어서 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=HAND_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")
    return cap


def extract_features(landmarks, handedness):
    """21개 랜드마크 -> 63차원 특징 벡터

    - 손목(0번) 기준 상대 좌표로 바꿔서 화면 위치에 상관없게 함
    - 손목~중지 MCP(9번) 거리로 나눠서 손 크기/카메라 거리에 상관없게 함
    - 왼손은 x를 뒤집어서 오른손과 같은 모양으로 맞춤 (한 손으로 수집해도 양손 인식)
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9, :2])
    if scale > 0:
        pts /= scale
    # 거울 모드 화면에서는 MediaPipe 라벨이 반대로 나오므로 "Right"가 사용자의 왼손
    if handedness[0].category_name == "Right":
        pts[:, 0] = -pts[:, 0]
    return pts.flatten()


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return pts
