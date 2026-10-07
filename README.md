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
- mediapipe 1.1.0, opencv-python 5.0.0, scikit-learn 1.9.1
- 웹: Chrome, MediaPipe Tasks JS `@mediapipe/tasks-vision@1.1.0`

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

### Face Landmarker 모델 구조

`face_landmarker.task`도 여러 모델이 묶인 번들입니다.

| 모델 | 입력 크기 | 역할 |
|---|---|---|
| FaceDetector | 192 x 192 | 얼굴 위치 찾기 |
| FaceMesh-V2 | 256 x 256 | 얼굴 랜드마크 478개 (홍채 포함) |
| Blendshape | 1 x 146 x 2 | 랜드마크 일부로 표정 점수 52개 계산 |

`output_face_blendshapes=True`를 줘야 `eyeBlinkLeft`, `jawOpen` 같은 표정 점수(0~1)가 나옵니다.

### 제스처 분류: 이미지 대신 랜드마크로 학습

이미지를 직접 학습하는 대신 Hand Landmarker가 뽑아 준 21개 점(63개 숫자)만 학습합니다. 입력이 작아서 제스처당 수백 개만 모아도 되고, 학습은 몇 초면 끝납니다.

같은 손 모양이면 같은 숫자가 나오도록 특징을 정규화합니다 (`gesture_common.extract_features`).

1. **위치**: 모든 점에서 손목(0번) 좌표를 빼서 화면 어디에 있든 같게
2. **크기**: 손목~중지 뿌리(9번) 거리로 나눠서 카메라와의 거리에 상관없게
3. **왼손/오른손**: 왼손이면 x 부호를 뒤집어 오른손 모양으로 맞춤

분류기는 `StandardScaler` + `MLPClassifier(64, 32)`입니다 (입력 63 → 64 → 32 → 클래스 수, ReLU, 출력 softmax).

학습할 때는 데이터의 20%를 떼어 평가한 뒤, 결과를 확인하고 나서 **전체 데이터로 다시 학습**해 저장합니다.

현재 데이터 결과 (`nike` 318개, `none` 315개, `ok` 319개 / 테스트 191개):

```
테스트 정확도: 0.974

혼동 행렬 (행=정답, 열=예측)
        nike  none  ok
nike      63     1   0
none       2    61   0
ok         1     1  62
```

`none`(아무 제스처도 아닌 손)을 따로 수집해 둔 것이 중요합니다. 없으면 손만 보여도 nike나 ok 중 하나로 억지로 분류됩니다.

### tkinter 앱을 만들 때 주의한 점

- **OpenCV는 한글을 못 그림**: `cv2.putText`의 기본 폰트는 영문만 지원해서 한글 제스처 이름이 깨집니다. 그래서 이름과 결과는 영상이 아니라 tkinter 라벨에 표시했습니다.
- **SPACE 키 중복 동작**: 버튼을 클릭하면 버튼에 포커스가 남아서, SPACE를 누르면 그 버튼도 눌리고 녹화도 토글됩니다. 버튼에 `takefocus=False`를 줘서 막았습니다.
- **이미지가 안 보이는 문제**: `ImageTk.PhotoImage`는 Python 변수로 참조를 유지하지 않으면 바로 지워집니다 (`self.video.image = img`).
- **학습 중 화면 멈춤 방지**: 학습은 별도 스레드에서 돌리고, 끝나면 `root.after()`로 UI를 갱신합니다 (tkinter는 메인 스레드에서만 UI를 바꿔야 함).

### Python 모델을 웹에서 돌리기

scikit-learn 모델은 브라우저에서 그대로 쓸 수 없어서 **가중치만 JSON으로 꺼내고 계산은 JavaScript로 직접** 구현했습니다.

```
x = (특징 - mean) / scale                 # StandardScaler
h = relu(x · W1 + b1)                      # 은닉층 1
h = relu(h · W2 + b2)                      # 은닉층 2
확률 = softmax(h · W3 + b3)                # 출력층
```

수집 데이터로 Python과 JavaScript 결과를 비교해서 확률 차이가 약 1e-7(float 오차 수준)임을 확인했습니다. 특징 추출 함수도 같은 값을 내는지 따로 비교했습니다.

웹과 Python이 같은 결과를 내려면 **입력 조건도 같아야** 합니다.

- **화면 비율**: 랜드마크 x, y는 이미지 폭·높이 기준 0~1 값이라 화면 비율이 다르면 손 모양 숫자가 찌그러집니다. Python 웹캠이 640x480(4:3)이라 웹에서도 영상을 4:3으로 잘라서 검출합니다.
- **거울 모드**: Python에서 뒤집은 프레임으로 학습했으므로 웹에서도 뒤집은 프레임으로 검출합니다 (Left/Right 라벨 의미도 같아짐).

그 외:

- 확률을 최근 5프레임 평균으로 써서 로고가 깜빡이지 않게 했습니다.
- `navigator.mediaDevices.getUserMedia`(웹캠)는 `https`나 `localhost`에서만 동작합니다.
- Windows에서는 웹캠을 한 프로그램만 쓸 수 있어서, Python 앱이 켜져 있으면 브라우저가 웹캠을 못 엽니다.

### GitHub Pages 배포

```bash
# 저장소 Settings > Pages 에서 해도 되고, gh CLI로는:
gh api -X POST repos/<owner>/<repo>/pages -f "source[branch]=main" -f "source[path]=/"
```

- 저장소 루트를 그대로 배포하므로 `web/app.js`가 `../hand_landmarker.task`로 모델을 불러올 수 있습니다.
- 루트 `index.html`은 `web/`으로 바로 넘겨 주는 페이지입니다.
- `.nojekyll`을 두면 GitHub Pages가 Jekyll 변환 없이 파일을 그대로 올립니다.
- push 후 1분 정도면 반영됩니다.

## 모델 다시 받기

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## 설정 변경

각 스크립트 상단의 상수로 조정할 수 있습니다.

- `CAMERA_INDEX`: 사용할 웹캠 번호 (기본 0)
- `NUM_HANDS` / `NUM_FACES`: 동시에 검출할 손·얼굴 수
