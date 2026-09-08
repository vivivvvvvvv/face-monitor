# 解决 PyTorch 2.6+ 兼容性问题
import torch
from numpy.core.multiarray import _reconstruct
torch.serialization.add_safe_globals([_reconstruct])

# 重载 torch.load 函数，自动添加 weights_only=False 参数
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import sys
sys.path.append('.')
# -*- coding: UTF-8 -*-
import time

import cv2
import torch
import copy
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from utils.torch_utils import time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)
    return model


def show_results(img, xywh, conf, landmarks, class_num, face_id):
    """绘制人脸框、关键点，并在框的左上角外侧显示序号（黑底绿字），不显示置信度"""
    h, w, c = img.shape
    tl = 2
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    # 关键点
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    radius = 2
    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), radius, clors[i], -1)

    # 序号（黑底绿字）
    label = str(face_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness_txt = 2
    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness_txt)
    padding = 3
    rect_x1 = x1 - padding
    rect_y1 = y1 - label_h - padding
    rect_x2 = x1 + label_w + padding
    rect_y2 = y1 + padding
    if rect_y1 < 0:
        rect_y1 = y1 + padding
        rect_y2 = y1 + label_h + 2 * padding
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.putText(img, label, (x1, rect_y1 + label_h), font, font_scale, (0, 255, 0), thickness_txt)

    return img


def detect_one(model, image_path, device):
    img_size = 640
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    print('pred: ', pred)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    results_list = []
    gain = min(img.shape[2] / orgimg.shape[0], img.shape[3] / orgimg.shape[1])
    pad = ((img.shape[3] - orgimg.shape[1] * gain) / 2, (img.shape[2] - orgimg.shape[0] * gain) / 2)

    face_counter = 0
    for i, det in enumerate(pred):
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            for j in range(det.size()[0]):
                face_counter += 1
                # 关键点坐标转换
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
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num, face_counter)

                # 疲劳分数
                left_eye_x, left_eye_y = landmarks[0], landmarks[1]
                right_eye_x, right_eye_y = landmarks[2], landmarks[3]
                eye_distance = np.sqrt((left_eye_x - right_eye_x)**2 + (left_eye_y - right_eye_y)**2)
                face_width = xywh[2] * orgimg.shape[1]
                fatigue_score = eye_distance / face_width if face_width > 0 else 0.0

                if fatigue_score < 0.18:
                    status = "Drowsy"
                elif fatigue_score < 0.22:
                    status = "Tired"
                else:
                    status = "Normal"

                results_list.append((fatigue_score, conf, status))

    print(f'Done. ({time.time() - t0:.3f}s)')

    # 在左上角顶格显示所有人脸的信息
    if results_list:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_txt = 1
        line_spacing = 5
        start_x = 0          # 顶格
        start_y = 20         # 第一个文字基线的y坐标（可调整，但背景会从0开始）
        padding = 5
        lines = [f"Face {idx+1}: Fatigue={fat:.3f} ({st}), Conf={c:.3f}" for idx, (fat, c, st) in enumerate(results_list)]
        # 计算最大宽度和总高度
        max_w = 0
        line_h = 0
        for line in lines:
            (w_line, _), _ = cv2.getTextSize(line, font, font_scale, thickness_txt)
            (_, h_line), _ = cv2.getTextSize(line, font, font_scale, thickness_txt)
            max_w = max(max_w, w_line)
            line_h = h_line
        rect_w = max_w + 2 * padding
        rect_h = len(lines) * (line_h + line_spacing) - line_spacing + 2 * padding
        # 绘制背景矩形：从 (0,0) 开始
        cv2.rectangle(orgimg, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
        # 绘制文字
        for i, line in enumerate(lines):
            y_pos = start_y + i * (line_h + line_spacing)
            cv2.putText(orgimg, line, (start_x + padding, y_pos), font, font_scale, (0, 255, 0), thickness_txt)
    else:
        cv2.putText(orgimg, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite('fatigue_result.jpg', orgimg)
    print("结果已保存为 fatigue_result.jpg")

    # 显示图片（缩放）
    h, w = orgimg.shape[:2]
    scale = min(1920/w, 1080/h)
    if scale < 1:
        new_w, new_h = int(w*scale), int(h*scale)
        display = cv2.resize(orgimg, (new_w, new_h))
    else:
        display = orgimg
    cv2.imshow('Result', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = 'yolov5s-face.pt'
    model = load_model(weights, device)
    image_path = '1.jpg'
    detect_one(model, image_path, device)
    print('over')