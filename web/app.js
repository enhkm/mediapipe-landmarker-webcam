// 웹캠 -> MediaPipe Hand Landmarker -> 학습한 MLP(gesture_model.json) -> 로고/이모지 표시
// MediaPipe JS와 wasm은 반드시 같은 버전이어야 함 (버전을 빼면 서로 다른 캐시 버전이 섞여 LinkError 발생)
// Python(gesture_common.py)과 같은 조건으로 추론하도록
//   1) 640x480(4:3) 거울 모드 프레임으로 검출하고
//   2) extract_features()와 같은 방식으로 특징을 만든다.
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WIDTH = 640, HEIGHT = 480;
const SMOOTH_FRAMES = 5; // 최근 N프레임 확률 평균 (깜빡임 방지)
// 제스처 이름 -> 화면에 띄울 요소
const OVERLAYS = { nike: document.getElementById("nikeLogo"), ok: document.getElementById("okEmoji") };

const video = document.getElementById("video");
const view = document.getElementById("view");
const ctx = view.getContext("2d");
const labelEl = document.getElementById("label");
const statusEl = document.getElementById("status");
const confEl = document.getElementById("conf");
const confText = document.getElementById("confText");
confEl.oninput = () => (confText.textContent = Number(confEl.value).toFixed(2));

// ---------- 분류기 (scikit-learn StandardScaler + MLPClassifier 재현) ----------
function predictProba(model, features) {
  let h = features.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  const last = model.coefs.length - 1;
  model.coefs.forEach((W, layer) => {
    const b = model.intercepts[layer];
    const out = b.slice();
    for (let i = 0; i < h.length; i++) {
      const row = W[i], x = h[i];
      for (let j = 0; j < out.length; j++) out[j] += x * row[j];
    }
    h = layer < last ? out.map(v => activate(model.activation, v)) : out;
  });
  if (model.out_activation === "softmax") {
    const m = Math.max(...h);
    const e = h.map(v => Math.exp(v - m));
    const s = e.reduce((a, c) => a + c, 0);
    return e.map(v => v / s);
  }
  const p = 1 / (1 + Math.exp(-h[0])); // 클래스 2개일 때 (logistic)
  return [1 - p, p];
}

function activate(kind, v) {
  if (kind === "relu") return Math.max(0, v);
  if (kind === "tanh") return Math.tanh(v);
  if (kind === "logistic") return 1 / (1 + Math.exp(-v));
  return v; // identity
}

// gesture_common.extract_features 와 동일
function extractFeatures(landmarks, handedness) {
  const w = landmarks[0];
  const pts = landmarks.map(p => [p.x - w.x, p.y - w.y, p.z - w.z]);
  const scale = Math.hypot(pts[9][0], pts[9][1]);
  const flip = handedness[0].categoryName === "Right"; // 거울 화면에서 "Right" = 사용자의 왼손
  return pts.flatMap(([x, y, z]) => {
    if (scale > 0) { x /= scale; y /= scale; z /= scale; }
    return [flip ? -x : x, y, z];
  });
}

// ---------- 화면 ----------
function buildBars(classes) {
  const bars = document.getElementById("bars");
  return classes.map(name => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span></span><div class="track"><div class="fill"></div></div><span class="num">0.00</span>`;
    row.children[0].textContent = name;
    bars.appendChild(row);
    return { fill: row.querySelector(".fill"), num: row.querySelector(".num") };
  });
}

function drawHand(landmarks) {
  ctx.strokeStyle = "#00ff00";
  ctx.lineWidth = 3;
  for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[start].x * WIDTH, landmarks[start].y * HEIGHT);
    ctx.lineTo(landmarks[end].x * WIDTH, landmarks[end].y * HEIGHT);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff3030";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * WIDTH, p.y * HEIGHT, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function showOverlay(name) {
  for (const [key, el] of Object.entries(OVERLAYS)) el.classList.toggle("show", key === name);
}

// 웹캠 영상을 4:3 가운데 잘라 거울 모드로 그림 (Python과 같은 640x480 프레임)
function drawMirroredFrame() {
  const vw = video.videoWidth, vh = video.videoHeight;
  let sw = vw, sh = vh;
  if (vw / vh > WIDTH / HEIGHT) sw = vh * WIDTH / HEIGHT; else sh = vw * HEIGHT / WIDTH;
  ctx.save();
  ctx.translate(WIDTH, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, (vw - sw) / 2, (vh - sh) / 2, sw, sh, 0, 0, WIDTH, HEIGHT);
  ctx.restore();
}

// ---------- 메인 ----------
async function main() {
  const [model, vision] = await Promise.all([
    fetch("gesture_model.json").then(r => {
      if (!r.ok) throw new Error("gesture_model.json 이 없습니다. gesture_train.py로 학습하세요.");
      return r.json();
    }),
    FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm"),
  ]);
  const landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "../hand_landmarker.task" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  const bars = buildBars(model.classes);

  statusEl.textContent = "웹캠 연결 중...";
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: WIDTH, height: HEIGHT } });
  await video.play();
  statusEl.textContent = `제스처: ${model.classes.join(", ")}`;

  const history = [];
  let lastTime = -1;
  function loop() {
    requestAnimationFrame(loop);
    if (video.currentTime === lastTime) return;
    lastTime = video.currentTime;

    drawMirroredFrame();
    const result = landmarker.detectForVideo(view, performance.now());

    if (!result.landmarks.length) {
      history.length = 0;
      labelEl.textContent = "손 없음";
      bars.forEach(b => { b.fill.style.width = "0"; b.num.textContent = "0.00"; });
      showOverlay(null);
      return;
    }

    drawHand(result.landmarks[0]);
    history.push(predictProba(model, extractFeatures(result.landmarks[0], result.handedness[0])));
    if (history.length > SMOOTH_FRAMES) history.shift();
    const proba = model.classes.map((_, i) => history.reduce((s, p) => s + p[i], 0) / history.length);

    proba.forEach((p, i) => {
      bars[i].fill.style.width = `${p * 100}%`;
      bars[i].num.textContent = p.toFixed(2);
    });
    const best = proba.indexOf(Math.max(...proba));
    const name = proba[best] >= Number(confEl.value) ? model.classes[best] : null;
    labelEl.textContent = name ?? "?";
    showOverlay(name);
  }
  loop();
}

main().catch(err => {
  console.error(err);
  statusEl.textContent = "오류: " + err.message;
});
