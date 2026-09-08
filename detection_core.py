# detection_core.py
# 核心检测逻辑，使用 MediaPipe 替代 dlib

import torch
import cv2
import numpy as np
import copy
import sys
import os
import mediapipe as mp

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from models.experimental import attempt_load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# ---------- MediaPipe 初始化 ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 左右眼关键点索引（MediaPipe 468 点模型）
LEFT_EYE_INDICES = [33, 133, 157, 159, 161, 173]
RIGHT_EYE_INDICES = [362, 263, 386, 384, 381, 380]

def eye_aspect_ratio(landmarks, indices):
    """计算 EAR"""
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def compute_ear_from_roi(face_roi):
    """在一个人脸 ROI 内用 MediaPipe 计算 EAR"""
    if face_roi is None or face_roi.size == 0:
        return None
    h, w = face_roi.shape[:2]
    if h < 30 or w < 30:
        return None
    rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_roi)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
    return (left_ear + right_ear) / 2.0

# ---------- 模型加载 ----------
def load_model(weights='yolov5s-face.pt'):
    global model
    if model is None:
        model = attempt_load(weights, map_location=device)
    return model

# ---------- 核心处理函数 ----------
def _process_image_array(orgimg):
    global model
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

    results = []
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

                # ---- 用 MediaPipe 计算 EAR ----
                ear = None
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
                if face_roi.size > 0:
                    ear = compute_ear_from_roi(face_roi)

                # ---- 状态判断 ----
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

                # ---- 绘图 ----
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
