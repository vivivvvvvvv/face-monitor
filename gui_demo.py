# -*- coding: UTF-8 -*-
import torch
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from numpy.core.multiarray import _reconstruct
torch.serialization.add_safe_globals([_reconstruct])

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import copy
import sys
import os
import tempfile
import shutil
import dlib
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from utils.torch_utils import time_synchronized
from models.experimental import attempt_load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# ---------- dlib ----------
dlib_predictor = None
dlib_available = False
LEFT_EYE_IDX = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def load_model(weights='yolov5s-face.pt'):
    global model
    if model is None:
        model = attempt_load(weights, map_location=device)
    return model

def process_image(img_path):
    global model
    if model is None:
        load_model()
    _, ext = os.path.splitext(img_path)
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_fd)
    shutil.copy2(img_path, temp_path)
    try:
        orgimg = cv2.imread(temp_path)
        if orgimg is None:
            raise ValueError(f"Cannot read image: {img_path}")
    finally:
        os.unlink(temp_path)
    return _process_image_array(orgimg)

def process_frame(frame):
    global model
    if model is None:
        load_model()
    return _process_image_array(frame)

def _process_image_array(orgimg):
    global model, dlib_predictor, dlib_available
    img_size = 640
    conf_thres = 0.3
    iou_thres = 0.5

    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    gain = min(img.shape[2] / orgimg.shape[0], img.shape[3] / orgimg.shape[1])
    pad = ((img.shape[3] - orgimg.shape[1] * gain) / 2, (img.shape[2] - orgimg.shape[0] * gain) / 2)

    results = []  # (face_counter, fatigue_display, eye_display, ear, area)
    face_counter = 0

    for det in pred:
        if len(det):
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            for j in range(det.size()[0]):
                face_counter += 1
                lks = det[j, 5:15].clone()
                lks[0::2] -= pad[0]
                lks[1::2] -= pad[1]
                lks[0::2] /= gain
                lks[1::2] /= gain
                lks[0::2].clamp_(0, orgimg.shape[1])
                lks[1::2].clamp_(0, orgimg.shape[0])
                landmarks = lks.cpu().numpy().tolist()

                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                area = xywh[2] * xywh[3]

                left_eye_x, left_eye_y = landmarks[0], landmarks[1]
                right_eye_x, right_eye_y = landmarks[2], landmarks[3]
                eye_dist = np.hypot(left_eye_x - right_eye_x, left_eye_y - right_eye_y)
                face_w = xywh[2] * orgimg.shape[1]
                fatigue_simple = eye_dist / face_w if face_w > 0 else 0.0

                ear = None
                if dlib_available and dlib_predictor is not None:
                    h, w = orgimg.shape[:2]
                    x1 = int(xywh[0]*w - 0.5*xywh[2]*w)
                    y1 = int(xywh[1]*h - 0.5*xywh[3]*h)
                    x2 = int(xywh[0]*w + 0.5*xywh[2]*w)
                    y2 = int(xywh[1]*h + 0.5*xywh[3]*h)

                    margin = 15
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)

                    face_roi = orgimg[y1:y2, x1:x2]
                    if face_roi.shape[0] >= 40 and face_roi.shape[1] >= 40:
                        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        gray_roi = np.ascontiguousarray(gray_roi, dtype=np.uint8)
                        dlib_rect = dlib.rectangle(0, 0, face_roi.shape[1], face_roi.shape[0])
                        try:
                            landmarks_68 = dlib_predictor(gray_roi, dlib_rect)
                            left_eye_pts = np.array([(landmarks_68.part(i).x, landmarks_68.part(i).y) for i in LEFT_EYE_IDX])
                            right_eye_pts = np.array([(landmarks_68.part(i).x, landmarks_68.part(i).y) for i in RIGHT_EYE_IDX])
                            left_ear = eye_aspect_ratio(left_eye_pts)
                            right_ear = eye_aspect_ratio(right_eye_pts)
                            ear = (left_ear + right_ear) / 2.0
                        except Exception:
                            ear = None

                if ear is not None:
                    if ear < 0.18:
                        fatigue_display = "😴 Drowsy"
                        eye_display = "🚫 Closed"
                    elif ear < 0.25:
                        fatigue_display = "😑 Tired"
                        eye_display = "👀 Half-closed"
                    else:
                        fatigue_display = "😊 Normal"
                        eye_display = "👁️ Open"
                else:
                    if fatigue_simple < 0.18:
                        fatigue_display = "😴 Drowsy"
                        eye_display = "🚫 Closed"
                    elif fatigue_simple < 0.22:
                        fatigue_display = "😑 Tired"
                        eye_display = "👀 Half-closed"
                    else:
                        fatigue_display = "😊 Normal"
                        eye_display = "👁️ Open"

                results.append((face_counter, fatigue_display, eye_display, ear, area))

                # 绘图
                h, w = orgimg.shape[:2]
                x1 = int(xywh[0]*w - 0.5*xywh[2]*w)
                y1 = int(xywh[1]*h - 0.5*xywh[3]*h)
                x2 = int(xywh[0]*w + 0.5*xywh[2]*w)
                y2 = int(xywh[1]*h + 0.5*xywh[3]*h)

                cv2.rectangle(orgimg, (x1, y1), (x2, y2), (0,255,0), 2)
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
                for i in range(5):
                    px = int(landmarks[2*i])
                    py = int(landmarks[2*i+1])
                    cv2.circle(orgimg, (px, py), 3, colors[i], -1)

                cv2.putText(orgimg, f"#{face_counter}", (x1+5, y1+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return orgimg, results


class FaceLandmarkGUI:
    def __init__(self, root):
        self.root = root
        root.title("Face and Landmark Detection (Fatigue + EAR + PERCLOS)")
        root.geometry("1200x900")

        # 按钮
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        self.detect_btn = tk.Button(btn_frame, text="Detect Image", command=self.detect_image, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        self.camera_btn = tk.Button(btn_frame, text="Start Camera", command=self.start_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        # PERCLOS 状态栏
        self.perclos_frame = tk.Frame(root, bg="#f0f0f0", relief=tk.RIDGE, bd=2)
        self.perclos_frame.pack(fill=tk.X, padx=10, pady=5)
        self.perclos_label = tk.Label(
            self.perclos_frame,
            text="PERCLOS: --% | Risk: -- | (Face --)",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        self.perclos_label.pack(pady=8)

        # 图片
        img_frame = tk.Frame(root)
        img_frame.pack(fill=tk.BOTH, expand=True)
        self.original_label = tk.Label(img_frame, text="Original Image", relief=tk.SUNKEN)
        self.original_label.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.result_label = tk.Label(img_frame, text="Result Image", relief=tk.SUNKEN)
        self.result_label.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # ---- 表格：6列 ----
        self.tree_frame = tk.Frame(root)
        self.tree_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        self.tree = ttk.Treeview(self.tree_frame, columns=("ID", "Fatigue", "PERCLOS", "Eye", "Head", "Action"), show="headings", height=6)
        self.tree.heading("ID", text="Face ID")
        self.tree.heading("Fatigue", text="Fatigue State")
        self.tree.heading("PERCLOS", text="PERCLOS")
        self.tree.heading("Eye", text="Eye State")
        self.tree.heading("Head", text="Head State")
        self.tree.heading("Action", text="Action")
        self.tree.column("ID", width=70, anchor="center")
        self.tree.column("Fatigue", width=150, anchor="center")
        self.tree.column("PERCLOS", width=90, anchor="center")
        self.tree.column("Eye", width=150, anchor="center")
        self.tree.column("Head", width=120, anchor="center")
        self.tree.column("Action", width=120, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # ---- PERCLOS：每个人有自己的队列 ----
        self.perclos_histories = {}  # face_id -> deque
        self.perclos_threshold = 0.18
        self.perclos_window = 150
        self.perclos_face_id = None
        self.perclos_value = 0.0

        self.image_path = None
        self.orig_img = None
        self.res_img = None
        self.cap = None
        self.camera_running = False

        self.root.after(100, self.init_model)

    def get_perclos_for_face(self, face_id, ear):
        if face_id not in self.perclos_histories:
            self.perclos_histories[face_id] = deque(maxlen=self.perclos_window)
        q = self.perclos_histories[face_id]
        if ear is not None:
            q.append(ear)
        if len(q) == 0:
            return 0.0
        closed = sum(1 for e in q if e < self.perclos_threshold)
        return (closed / len(q)) * 100.0

    def update_perclos_display(self):
        if self.perclos_value < 0 or self.perclos_face_id is None:
            self.perclos_label.config(text="PERCLOS: --% | Risk: -- | (Face --)")
            return
        text = f"PERCLOS: {self.perclos_value:.1f}% | "
        if self.perclos_value < 10:
            text += "Risk: 😊 Normal"
            color = "#27ae60"
        elif self.perclos_value < 30:
            text += "Risk: 😑 Mild Fatigue"
            color = "#f39c12"
        else:
            text += "Risk: 😴 High Fatigue"
            color = "#e74c3c"
        text += f" | (Face #{self.perclos_face_id})"
        self.perclos_label.config(text=text, fg=color)

    def init_model(self):
        global dlib_predictor, dlib_available
        try:
            load_model()
            self.insert_tree("System", "YOLO model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            self.root.destroy()
            return

        dat_file = 'shape_predictor_68_face_landmarks.dat'
        if not os.path.exists(dat_file):
            self.insert_tree("System", "Warning: dlib predictor file not found. EAR disabled.")
            dlib_available = False
            dlib_predictor = None
            return
        try:
            dlib_predictor = dlib.shape_predictor(dat_file)
            dlib_available = True
            self.insert_tree("System", "dlib predictor loaded successfully.")
        except Exception as e:
            self.insert_tree("System", f"Warning: Failed to load dlib predictor: {e}. EAR disabled.")
            dlib_available = False
            dlib_predictor = None

    def insert_tree(self, col1, col2, col3=None, col4=None, col5="—", col6="—"):
        """
        插入一行数据
        - 系统消息：insert_tree("System", "YOLO loaded") → 合并显示在第一列
        - 人脸数据：insert_tree(face_id, fatigue, perclos_val, eye, head, action)
        """
        # 系统消息：只有 2 个参数，或者第一个参数是 "System" / "No face"
        if isinstance(col1, str) and col2 and not isinstance(col2, dict):
            if col3 is None and col4 is None:
                self.tree.insert("", 0, values=(f"{col1} {col2}", "", "", "", "", ""))
                return
            if col1 in ["No face", "System"]:
                self.tree.insert("", 0, values=(f"{col1} {col2}", "", "", "", "", ""))
                return

        # 正常数据：5 个参数
        perclos_str = f"{col3:.1f}%" if isinstance(col3, (int, float)) and col3 >= 0 else "--"
        self.tree.insert("", 0, values=(f"#{col1}", col2, perclos_str, col4, col5, col6))

        if len(self.tree.get_children()) > 20:
            last = self.tree.get_children()[-1]
            self.tree.delete(last)

    def clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.image_path = path
        try:
            img = Image.open(path)
            img.thumbnail((600, 600), Image.LANCZOS)
            self.orig_img = ImageTk.PhotoImage(img)
            self.original_label.config(image=self.orig_img)
            self.original_label.image = self.orig_img
            self.detect_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image: {e}")

    def detect_image(self):
        if not self.image_path:
            return
        try:
            result_img, results = process_image(self.image_path)
            res_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)
            res_pil.thumbnail((600, 600), Image.LANCZOS)
            self.res_img = ImageTk.PhotoImage(res_pil)
            self.result_label.config(image=self.res_img)
            self.result_label.image = self.res_img

            self.perclos_histories.clear()
            self.perclos_value = -1
            self.perclos_face_id = None
            self.update_perclos_display()

            self.clear_tree()
            if not results:
                self.insert_tree("No face", "detected")
                return

            sorted_results = sorted(results, key=lambda x: x[4], reverse=True)
            for face_id, fatigue_display, eye_display, ear, area in sorted_results:
                self.insert_tree(face_id, fatigue_display, -1, eye_display, "—", "—")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_camera(self):
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        self.camera_running = True
        self.camera_btn.config(text="Stop Camera", command=self.stop_camera)
        self.perclos_histories.clear()
        self.perclos_value = 0.0
        self.perclos_face_id = None
        self.update_perclos_display()
        self.clear_tree()
        self.update_camera()

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_btn.config(text="Start Camera", command=self.start_camera)
        self.original_label.config(image='')
        self.result_label.config(image='')
        self.orig_img = None
        self.res_img = None
        self.clear_tree()
        self.perclos_histories.clear()
        self.perclos_value = -1
        self.perclos_face_id = None
        self.update_perclos_display()

    def update_camera(self):
        if not self.camera_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            result_img, results = process_frame(frame)

            sorted_results = sorted(results, key=lambda x: x[4], reverse=True)

            # 更新每张脸的 PERCLOS
            perclos_map = {}
            for face_id, _, _, ear, _ in sorted_results:
                perclos_map[face_id] = self.get_perclos_for_face(face_id, ear)

            # 更新最大脸的状态栏
            if sorted_results and len(sorted_results) > 0:
                top_face_id = sorted_results[0][0]
                self.perclos_face_id = top_face_id
                self.perclos_value = perclos_map.get(top_face_id, 0.0)
            else:
                self.perclos_face_id = None
                self.perclos_value = -1
            self.update_perclos_display()

            # 显示图片
            res_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)
            res_pil.thumbnail((600, 600), Image.LANCZOS)
            self.res_img = ImageTk.PhotoImage(res_pil)
            self.result_label.config(image=self.res_img)
            self.result_label.image = self.res_img

            original_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            original_pil.thumbnail((600, 600), Image.LANCZOS)
            self.orig_img = ImageTk.PhotoImage(original_pil)
            self.original_label.config(image=self.orig_img)
            self.original_label.image = self.orig_img

            # 更新表格
            self.clear_tree()
            if not sorted_results:
                self.insert_tree("No face", "detected")
            else:
                for face_id, fatigue_display, eye_display, ear, area in sorted_results:
                    perclos_val = perclos_map.get(face_id, 0.0)
                    self.insert_tree(face_id, fatigue_display, perclos_val, eye_display, "—", "—")

        self.root.after(30, self.update_camera)

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceLandmarkGUI(root)
    root.mainloop()