"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크 검출

필요:
  pip install mediapipe opencv-python
  face_landmarker.task 모델 파일을 이 스크립트와 같은 폴더에 둘 것
  (https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

종료: q 또는 ESC
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
CAMERA_INDEX = 0
NUM_FACES = 1

# 얼굴 메시(478개 랜드마크) 연결 인덱스 쌍
C = vision.FaceLandmarksConnections
TESSELATION = [(c.start, c.end) for c in C.FACE_LANDMARKS_TESSELATION]
CONTOURS = [(c.start, c.end) for c in C.FACE_LANDMARKS_CONTOURS]
IRISES = [(c.start, c.end) for c in C.FACE_LANDMARKS_LEFT_IRIS + C.FACE_LANDMARKS_RIGHT_IRIS]


def draw_faces(frame, result):
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for a, b in TESSELATION:
            cv2.line(frame, pts[a], pts[b], (90, 90, 90), 1)
        for a, b in CONTOURS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 1)
        for a, b in IRISES:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 1)

    # 첫 번째 얼굴의 표정(blendshape) 상위 5개 표시
    if result.face_blendshapes:
        top = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)[:5]
        for i, cat in enumerate(top):
            cv2.putText(frame, f"{cat.category_name} {cat.score:.2f}", (10, 60 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 여므로 바이트로 직접 읽어서 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못해 종료합니다.")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_faces(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  faces {len(result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
