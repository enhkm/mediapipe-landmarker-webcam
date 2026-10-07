# MediaPipe Landmarker Webcam

웹캠 영상에서 **손 랜드마크**와 **얼굴 랜드마크**를 실시간으로 검출하는 Python 예제입니다.
Google [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)의 Hand Landmarker / Face Landmarker를 사용합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `hand_landmarker_webcam.py` | 손 랜드마크 검출 (최대 2손, 21개 점) |
| `face_landmarker_webcam.py` | 얼굴 랜드마크 검출 (478개 점 + 표정 blendshape) |
| `hand_landmarker.task` | 손 모델 (float16, 7.8MB) |
| `face_landmarker.task` | 얼굴 모델 (float16, 3.7MB) |

모델 파일은 Google 공식 배포본이며 저장소에 포함되어 있어 clone 후 바로 실행할 수 있습니다.

## 테스트 환경

- Windows 11, Python 3.14
- mediapipe 1.1.0, opencv-python 5.0.0

## 설치 및 실행

```bash
pip install mediapipe opencv-python

python hand_landmarker_webcam.py
python face_landmarker_webcam.py
```

종료: `q` 또는 `ESC`

## 화면 표시 내용

**손 (Hand Landmarker)**
- 손 뼈대(초록 선)와 관절 점(빨간 점)
- 손마다 Left/Right 라벨과 신뢰도
- 좌상단에 FPS, 검출된 손 개수

**얼굴 (Face Landmarker)**
- 얼굴 메시(회색), 눈·눈썹·입술·윤곽선(초록), 홍채(주황)
- 좌상단에 FPS, 검출된 얼굴 수, 점수가 높은 표정 blendshape 5개 (예: `eyeBlinkLeft`, `jawOpen`)

두 스크립트 모두 거울 모드(좌우 반전)로 표시하며, 웹캠 프레임을 읽지 못하면 메시지를 출력하고 종료합니다.

## 주의: 한글 경로 문제 (Windows)

Windows에서 폴더 경로에 한글이 있으면 MediaPipe가 `model_asset_path`로 모델을 열지 못합니다.

```
RuntimeError: Unable to open file at C:\...\�ǽ�\hand_landmarker.task
```

그래서 두 스크립트 모두 모델 파일을 Python에서 바이트로 읽어 전달합니다.

```python
base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes())
```

## 모델 다시 받기

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## 설정 변경

각 스크립트 상단의 상수로 조정할 수 있습니다.

- `CAMERA_INDEX`: 사용할 웹캠 번호 (기본 0)
- `NUM_HANDS` / `NUM_FACES`: 동시에 검출할 손·얼굴 수
