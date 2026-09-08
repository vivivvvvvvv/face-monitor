# detection_core.py
# 轻量版：只用 YOLOv5-face 五点检测 + 简化疲劳分数（眼距/脸宽）
# 无 dlib / MediaPipe 依赖，适合 Render 512MB 环境

import torch
import cv2
import numpy as np
import copy
import sys
import os

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from models.experimental import attempt_load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model(weights='yolov5n-0.5.pt'):   # 使用小模型
    global model
    if model is None:
        model = attempt_load(weights, map_location=device)
    return model

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

                # ---- 简化疲劳分数（眼距 / 脸宽） ----
                left_eye_x, left_eye_y = landmarks[0], landmarks[1]
                right_eye_x, right_eye_y = landmarks[2], landmarks[3]
                eye_dist = np.hypot(left_eye_x - right_eye_x, left_eye_y - right_eye_y)
                face_w = xywh[2] * orgimg.shape[1]
                fatigue_simple = eye_dist / face_w if face_w > 0 else 0.0

                # ---- 状态判断 ----
                if fatigue_simple < 0.18:
                    fatigue_display = "😴 Drowsy"
                    eye_display = "🚫 Closed (approx)"
                elif fatigue_simple < 0.22:
                    fatigue_display = "😑 Tired"
                    eye_display = "👀 Half-closed (approx)"
                else:
                    fatigue_display = "😊 Normal"
                    eye_display = "👁️ Open"

                # ear 设为 None，前端显示 N/A
                results.append((face_counter, fatigue_display, eye_display, None, area))

                # ---- 绘图 ----
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
