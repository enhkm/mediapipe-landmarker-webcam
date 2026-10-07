"""제스처 학습 앱 - 수집 / 학습 / 인식을 한 창에서

실행: python gesture_app.py

1. 오른쪽에 제스처 이름을 입력하고 [추가]
2. 목록에서 제스처를 고르고 [녹화] (또는 SPACE) 로 손 모양 수집
3. 제스처를 2개 이상 모았으면 [학습]
4. [인식] 모드로 바꿔서 실시간 확인
"""
import csv
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk

import cv2
import joblib
import mediapipe as mp
from PIL import Image, ImageTk

import gesture_train
from gesture_common import DATA_PATH, GESTURE_MODEL_PATH, create_landmarker, draw_hand, extract_features, open_camera

VIDEO_WIDTH = 640


class GestureApp:
    def __init__(self, root):
        self.root = root
        root.title("제스처 학습")

        self.counts = self.load_counts()
        self.labels = sorted(self.counts)
        self.recording = False
        self.training = False
        self.model = joblib.load(GESTURE_MODEL_PATH) if GESTURE_MODEL_PATH.exists() else None

        self.build_ui()
        self.refresh_list()

        self.cap = open_camera()
        self.landmarker = create_landmarker()
        self.start = time.monotonic()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", self.on_space)
        self.update_frame()

    # ---------- UI ----------
    def build_ui(self):
        self.video = ttk.Label(self.root)
        self.video.grid(row=0, column=0, padx=8, pady=8, sticky="n")

        panel = ttk.Frame(self.root, padding=8)
        panel.grid(row=0, column=1, sticky="ns")

        self.mode = tk.StringVar(value="collect")
        mode_box = ttk.LabelFrame(panel, text="모드", padding=6)
        mode_box.pack(fill="x")
        ttk.Radiobutton(mode_box, text="수집", value="collect", variable=self.mode, takefocus=False,
                        command=self.on_mode).pack(side="left", padx=4)
        ttk.Radiobutton(mode_box, text="인식", value="infer", variable=self.mode, takefocus=False,
                        command=self.on_mode).pack(side="left", padx=4)

        # 수집
        collect = ttk.LabelFrame(panel, text="제스처 (이름 / 샘플 수)", padding=6)
        collect.pack(fill="both", expand=True, pady=6)
        self.listbox = tk.Listbox(collect, height=8, exportselection=False, font=("Malgun Gothic", 11))
        self.listbox.pack(fill="both", expand=True)

        add_row = ttk.Frame(collect)
        add_row.pack(fill="x", pady=4)
        self.name_entry = ttk.Entry(add_row)
        self.name_entry.pack(side="left", fill="x", expand=True)
        self.name_entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(add_row, text="추가", width=6, takefocus=False, command=self.add_label).pack(side="left", padx=(4, 0))

        btn_row = ttk.Frame(collect)
        btn_row.pack(fill="x")
        self.rec_btn = ttk.Button(btn_row, text="● 녹화 (SPACE)", takefocus=False, command=self.toggle_record)
        self.rec_btn.pack(side="left", fill="x", expand=True)
        ttk.Button(btn_row, text="삭제", width=6, takefocus=False, command=self.delete_label).pack(side="left", padx=(4, 0))

        # 학습
        self.train_btn = ttk.Button(panel, text="학습", takefocus=False, command=self.start_training)
        self.train_btn.pack(fill="x")

        # 인식
        infer = ttk.LabelFrame(panel, text="인식 결과", padding=6)
        infer.pack(fill="x", pady=6)
        self.result_var = tk.StringVar(value="-")
        ttk.Label(infer, textvariable=self.result_var, font=("Malgun Gothic", 22, "bold"),
                  anchor="center").pack(fill="x")
        self.conf_var = tk.DoubleVar(value=0.7)
        conf_row = ttk.Frame(infer)
        conf_row.pack(fill="x")
        ttk.Label(conf_row, text="최소 확률").pack(side="left")
        ttk.Scale(conf_row, from_=0.3, to=0.99, variable=self.conf_var).pack(side="left", fill="x", expand=True)

        self.status_var = tk.StringVar()
        ttk.Label(panel, textvariable=self.status_var, foreground="#555", wraplength=260).pack(fill="x")

        self.log = tk.Text(self.root, height=12, font=("Consolas", 9))
        self.log.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=(0, 8))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.set_status("모델 있음 - 인식 모드 사용 가능" if self.model else "제스처를 추가하고 수집하세요")

    def set_status(self, text):
        self.status_var.set(text)

    def write_log(self, text):
        self.log.delete("1.0", "end")
        self.log.insert("end", text)

    def refresh_list(self, keep=None):
        keep = keep if keep is not None else self.selected_label()
        self.listbox.delete(0, "end")
        for name in self.labels:
            self.listbox.insert("end", f"{name}  ({self.counts[name]})")
        if keep in self.labels:
            i = self.labels.index(keep)
            self.listbox.selection_set(i)
            self.listbox.see(i)

    def selected_label(self):
        sel = self.listbox.curselection()
        return self.labels[sel[0]] if sel else None

    # ---------- 데이터 ----------
    @staticmethod
    def load_counts():
        if not DATA_PATH.exists():
            return Counter()
        with DATA_PATH.open(newline="", encoding="utf-8") as f:
            return Counter(row[0] for row in csv.reader(f) if row)

    def add_label(self):
        name = self.name_entry.get().strip()
        if not name or "," in name:
            self.set_status("이름을 입력하세요 (쉼표 제외)")
            return
        if name not in self.labels:
            self.labels.append(name)
        self.name_entry.delete(0, "end")
        self.refresh_list(keep=name)
        self.root.focus_set()  # SPACE가 입력창이 아니라 녹화로 가도록
        self.set_status(f"'{name}' 선택됨 - 손 모양을 만들고 녹화하세요")

    def delete_label(self):
        name = self.selected_label()
        if not name:
            return
        if self.counts[name] and not messagebox.askyesno(
                "삭제", f"'{name}' 샘플 {self.counts[name]}개를 삭제할까요?"):
            return
        self.stop_record()
        if DATA_PATH.exists():
            with DATA_PATH.open(newline="", encoding="utf-8") as f:
                rows = [row for row in csv.reader(f) if row and row[0] != name]
            with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
        self.labels.remove(name)
        del self.counts[name]
        self.refresh_list()
        self.set_status(f"'{name}' 삭제됨 - 다시 학습해야 모델에 반영됩니다")

    # ---------- 녹화 ----------
    def on_space(self, event):
        if self.root.focus_get() is self.name_entry:
            return
        self.toggle_record()

    def toggle_record(self):
        if self.recording:
            self.stop_record()
            return
        if self.mode.get() != "collect" or self.training:
            return
        name = self.selected_label()
        if not name:
            self.set_status("먼저 목록에서 제스처를 선택하세요")
            return
        self.recording = True
        self.rec_label = name
        self.rec_file = DATA_PATH.open("a", newline="", encoding="utf-8")
        self.rec_writer = csv.writer(self.rec_file)
        self.rec_btn.config(text="■ 정지 (SPACE)")
        self.listbox.config(state="disabled")
        self.set_status(f"'{name}' 녹화 중... 손 각도와 위치를 조금씩 바꿔 주세요")

    def stop_record(self):
        if not self.recording:
            return
        self.recording = False
        self.rec_file.close()
        self.rec_btn.config(text="● 녹화 (SPACE)")
        self.listbox.config(state="normal")
        self.refresh_list(keep=self.rec_label)
        self.set_status(f"'{self.rec_label}' {self.counts[self.rec_label]}개 저장됨")

    # ---------- 학습 ----------
    def start_training(self):
        if self.training:
            return
        self.stop_record()
        self.training = True
        self.train_btn.config(state="disabled", text="학습 중...")
        self.set_status("학습 중...")
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        try:
            model, report = gesture_train.train()
            self.root.after(0, self.finish_training, model, report, None)
        except Exception as e:
            self.root.after(0, self.finish_training, None, None, str(e))

    def finish_training(self, model, report, error):
        self.training = False
        self.train_btn.config(state="normal", text="학습")
        if error:
            self.set_status("학습 실패: " + error)
            return
        self.model = model
        self.write_log(report)
        self.set_status("학습 완료 - 인식 모드로 확인해 보세요")

    def on_mode(self):
        self.stop_record()
        if self.mode.get() == "infer" and self.model is None:
            self.set_status("학습된 모델이 없습니다. 먼저 [학습]을 누르세요")
        self.result_var.set("-")

    # ---------- 카메라 루프 ----------
    def update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, int((time.monotonic() - self.start) * 1000))
            hand = result.hand_landmarks[0] if result.hand_landmarks else None

            if hand:
                draw_hand(frame, hand)
                features = extract_features(hand, result.handedness[0])
                if self.recording:
                    self.rec_writer.writerow([self.rec_label, *(f"{v:.5f}" for v in features)])
                    self.counts[self.rec_label] += 1
                    i = self.labels.index(self.rec_label)
                    self.listbox.config(state="normal")
                    self.listbox.delete(i)
                    self.listbox.insert(i, f"{self.rec_label}  ({self.counts[self.rec_label]})")
                    self.listbox.config(state="disabled")
                elif self.mode.get() == "infer" and self.model is not None:
                    proba = self.model.predict_proba([features])[0]
                    best = proba.argmax()
                    name = str(self.model.classes_[best]) if proba[best] >= self.conf_var.get() else "?"
                    self.result_var.set(f"{name}  {proba[best]:.2f}")
            elif self.mode.get() == "infer":
                self.result_var.set("손 없음")

            if self.recording:
                cv2.circle(frame, (25, 25), 10, (0, 0, 255), -1)

            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (VIDEO_WIDTH, int(h * VIDEO_WIDTH / w)))
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video.config(image=img)
            self.video.image = img  # 참조를 유지하지 않으면 이미지가 사라짐
        self.root.after(10, self.update_frame)

    def close(self):
        self.stop_record()
        self.cap.release()
        self.landmarker.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    GestureApp(root)
    root.mainloop()
