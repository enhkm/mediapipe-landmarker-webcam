"""제스처 분류기 훈련 - gesture_data.csv로 학습해서 gesture_model.joblib 저장

gesture_app.py의 [학습] 버튼이 이 파일의 train()을 사용합니다.
UI 없이 학습만 하려면: python gesture_train.py
"""
import csv
import json
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_common import DATA_PATH, GESTURE_MODEL_PATH, WEB_MODEL_PATH

MIN_SAMPLES = 10


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("수집 데이터가 없습니다. 먼저 제스처를 수집하세요.")
    labels, features = [], []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                labels.append(row[0])
                features.append([float(v) for v in row[1:]])
    return np.array(features, dtype=np.float32), np.array(labels)


def export_web_model(model):
    """웹 버전(web/app.js)이 쓸 수 있도록 스케일러와 MLP 가중치를 JSON으로 저장"""
    scaler, mlp = model[0], model[-1]
    data = {
        "classes": [str(c) for c in model.classes_],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "activation": mlp.activation,
        "out_activation": mlp.out_activation_,
        "coefs": [w.tolist() for w in mlp.coefs_],
        "intercepts": [b.tolist() for b in mlp.intercepts_],
    }
    WEB_MODEL_PATH.parent.mkdir(exist_ok=True)
    WEB_MODEL_PATH.write_text(json.dumps(data), encoding="utf-8")


def train():
    """학습 후 모델을 저장하고 (model, 결과 리포트 문자열)을 반환"""
    X, y = load_data()
    counts = Counter(y)
    if len(counts) < 2:
        raise ValueError("제스처가 2개 이상 필요합니다.")
    few = [name for name, n in counts.items() if n < MIN_SAMPLES]
    if few:
        raise ValueError(f"제스처마다 최소 {MIN_SAMPLES}개 이상 수집하세요. 부족: {', '.join(few)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=42),
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    classes = [str(c) for c in model.classes_]
    lines = [
        "샘플 수: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        f"테스트 정확도: {(pred == y_test).mean():.3f}",
        "",
        classification_report(y_test, pred, zero_division=0),
        "혼동 행렬 (행=정답, 열=예측), 순서: " + ", ".join(classes),
        str(confusion_matrix(y_test, pred, labels=model.classes_)),
    ]

    # 평가가 끝났으면 전체 데이터로 다시 학습해서 저장
    model.fit(X, y)
    joblib.dump(model, GESTURE_MODEL_PATH)
    export_web_model(model)
    lines.append(f"\n모델 저장: {GESTURE_MODEL_PATH.name}, web/{WEB_MODEL_PATH.name}")
    return model, "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--export-only" in sys.argv:
        # 이미 학습된 gesture_model.joblib을 웹용 JSON으로만 변환
        export_web_model(joblib.load(GESTURE_MODEL_PATH))
        print("웹 모델 저장:", WEB_MODEL_PATH)
    else:
        print(train()[1])
