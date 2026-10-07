# MediaPipe Landmarker Webcam

웹캠 영상에서 **손 랜드마크**와 **얼굴 랜드마크**를 실시간으로 검출하는 Python 예제입니다.
Google [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)의 Hand Landmarker / Face Landmarker를 사용합니다.

손 랜드마크로 **나만의 제스처를 학습**하고, 학습한 모델을 **웹에서 실행**하는 예제도 포함되어 있습니다.

**웹 데모: https://enhkm.github.io/mediapipe-landmarker-webcam/** (웹캠 허용 필요, `nike` 제스처 → 스우시 로고, `ok` 제스처 → 👌)

## 파일 구성

| 파일 | 설명 |
|---|---|
| `hand_landmarker_webcam.py` | 손 랜드마크 검출 (최대 2손, 21개 점) |
| `face_landmarker_webcam.py` | 얼굴 랜드마크 검출 (478개 점 + 표정 blendshape) |
| `hand_landmarker.task` | 손 모델 (float16, 7.8MB) |
| `face_landmarker.task` | 얼굴 모델 (float16, 3.7MB) |
| `gesture_app.py` | 제스처 수집 / 학습 / 인식 UI 앱 (tkinter) |
| `gesture_train.py` | 제스처 분류기 학습 (앱의 [학습] 버튼이 사용) |
| `gesture_common.py` | 제스처 앱/학습 공용 코드 (특징 추출 등) |
| `gesture_data.csv` | 수집한 제스처 데이터 (nike / none / ok) |
| `gesture_model.joblib` | 학습된 제스처 분류기 |
| `web/` | 웹 버전 (GitHub Pages로 배포) |

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
- 손마다 Left/Right 라벨과 신뢰도 (사용자 기준 왼손/오른손)
- 좌상단에 FPS, 검출된 손 개수

**얼굴 (Face Landmarker)**
- 얼굴 메시(회색), 눈·눈썹·입술·윤곽선(초록), 홍채(주황)
- 좌상단에 FPS, 검출된 얼굴 수, 점수가 높은 표정 blendshape 5개 (예: `eyeBlinkLeft`, `jawOpen`)

두 스크립트 모두 거울 모드(좌우 반전)로 표시하며, 웹캠 프레임을 읽지 못하면 메시지를 출력하고 종료합니다.

## 나만의 제스처 학습

```bash
pip install mediapipe opencv-python scikit-learn pillow
python gesture_app.py
```

1. 오른쪽 입력창에 제스처 이름을 쓰고 **[추가]**
2. 손 모양을 만들고 **SPACE** (또는 [녹화])로 수집. 제스처당 100~300개 정도, 각도·거리를 조금씩 바꿔 가며
3. 제스처를 2개 이상 모았으면 **[학습]** → 아래에 정확도와 혼동 행렬이 표시됨
4. 모드를 **인식**으로 바꿔 실시간 확인

손 21개 점을 손목 기준 상대 좌표로 바꾸고 손 크기로 나눠서 화면 위치·거리에 상관없게 만든 뒤, 63차원 벡터를 작은 MLP(scikit-learn)로 분류합니다. 왼손은 x를 뒤집어 오른손과 같은 모양으로 맞추므로 한 손으로만 수집해도 양손 모두 인식됩니다.

아무 제스처도 아닐 때를 위해 `none` 같은 "기타" 제스처도 함께 수집해 두면 오인식이 줄어듭니다.

## 웹 버전 (GitHub Pages)

`web/`은 브라우저에서 MediaPipe Tasks JS로 손을 검출하고, Python에서 학습한 모델을 JavaScript로 그대로 계산합니다.

- 학습할 때 `web/gesture_model.json` (스케일러 + MLP 가중치)도 함께 저장되므로, 다시 학습하고 push하면 웹에도 반영됩니다.
- Python과 같은 조건이 되도록 640x480(4:3) 거울 모드 프레임으로 검출합니다.
- MediaPipe JS와 wasm은 반드시 같은 버전(`@1.1.0`)으로 불러야 합니다. 버전을 빼면 서로 다른 버전이 섞여 `LinkError`가 납니다.

로컬에서 실행:

```bash
python -m http.server 8000
# http://127.0.0.1:8000/web/ 접속
```

웹캠은 `https` 또는 `localhost`에서만 쓸 수 있어서 파일을 더블클릭으로 열면 동작하지 않습니다.

## 주의: 한글 경로 문제 (Windows)

Windows에서 폴더 경로에 한글이 있으면 MediaPipe가 `model_asset_path`로 모델을 열지 못합니다.

```
RuntimeError: Unable to open file at C:\...\�ǽ�\hand_landmarker.task
```

그래서 두 스크립트 모두 모델 파일을 Python에서 바이트로 읽어 전달합니다.

```python
base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes())
```

## 공부한 내용 정리

### Hand Landmarker 모델 구조

`hand_landmarker.task`는 모델 2개가 묶인 번들입니다.

1. **손바닥 검출 모델 (palm detection)**: 이미지에서 손 위치를 찾음
2. **손 랜드마크 모델 (hand landmarks detection)**: 찾은 손 영역에서 21개 관절 좌표를 계산

| 항목 | 값 |
|---|---|
| 입력 크기 | 192 x 192, 224 x 224 |
| 양자화 | float16 |
| 학습 데이터 | 실제 이미지 약 3만 장 + 합성 손 모델 이미지 |

`VIDEO`/`LIVE_STREAM` 모드에서는 매 프레임마다 손바닥 검출을 하지 않습니다. 랜드마크 모델이 손을 놓치면(신뢰도가 `min_tracking_confidence`보다 낮으면) 그때 다시 검출합니다. 그래서 `IMAGE` 모드보다 빠릅니다.

### 21개 손 랜드마크 번호

| 번호 | 이름 | 번호 | 이름 | 번호 | 이름 |
|---|---|---|---|---|---|
| 0 | WRIST | 7 | INDEX_FINGER_DIP | 14 | RING_FINGER_PIP |
| 1 | THUMB_CMC | 8 | INDEX_FINGER_TIP | 15 | RING_FINGER_DIP |
| 2 | THUMB_MCP | 9 | MIDDLE_FINGER_MCP | 16 | RING_FINGER_TIP |
| 3 | THUMB_IP | 10 | MIDDLE_FINGER_PIP | 17 | PINKY_MCP |
| 4 | THUMB_TIP | 11 | MIDDLE_FINGER_DIP | 18 | PINKY_PIP |
| 5 | INDEX_FINGER_MCP | 12 | MIDDLE_FINGER_TIP | 19 | PINKY_DIP |
| 6 | INDEX_FINGER_PIP | 13 | RING_FINGER_MCP | 20 | PINKY_TIP |

손가락 끝은 4, 8, 12, 16, 20번입니다. 좌표 `x`, `y`는 이미지 폭·높이 기준 0~1로 정규화된 값이라 픽셀 좌표로 쓰려면 `x * w`, `y * h`로 바꿔야 합니다.

### 실행 모드와 타임스탬프

`RunningMode.VIDEO`에서는 `detect_for_video(image, timestamp_ms)`를 호출하며, 타임스탬프는 **계속 증가**해야 합니다. 그래서 `time.monotonic()`으로 시작 시점부터 지난 시간을 ms 단위로 넘깁니다.

### Left/Right 라벨이 반대로 나오는 문제

웹캠 화면을 거울 모드(`cv2.flip(frame, 1)`)로 뒤집은 뒤 검출했더니 MediaPipe의 `handedness` 라벨이 사용자 기준과 반대로 나왔습니다 (오른손을 들면 `Left`). 그래서 화면에 표시할 때 라벨을 바꿔서 출력합니다.

```python
name = {"Left": "Right", "Right": "Left"}[handedness[0].category_name]
```

### 실행 시 나오는 경고 로그

실행하면 아래와 같은 로그가 나오지만 MediaPipe 내부 메시지라 검출 결과에는 영향이 없습니다.

```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 ... inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. ...
W0000 ... landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. ...
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
