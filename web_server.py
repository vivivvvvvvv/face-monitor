# -*- coding: utf-8 -*-
import sys
import os
import base64
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# 导入你的核心检测函数
from gui_demo import _process_image_array, load_model
import gui_demo

# ---- 手动初始化 dlib（因为 gui_demo 的 GUI 类没有被实例化） ----
import dlib
dat_file = 'shape_predictor_68_face_landmarks.dat'
if os.path.exists(dat_file):
    try:
        gui_demo.dlib_predictor = dlib.shape_predictor(dat_file)
        gui_demo.dlib_available = True
        print("✅ dlib predictor loaded successfully in web_server")
    except Exception as e:
        print(f"❌ Failed to load dlib: {e}")
        gui_demo.dlib_available = False
else:
    print(f"❌ {dat_file} not found")
    gui_demo.dlib_available = False
# ---------------------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

print("⏳ 正在加载 YOLO 模型...")
load_model()
print("✅ YOLO 模型加载完成，Web 服务器准备就绪。")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('✅ 客户端已连接')
    emit('ready', {'status': 'connected'})

@socketio.on('image')
def handle_image(data):
    try:
        raw = data['image']
        if ',' in raw:
            raw = raw.split(',')[1]

        img_bytes = base64.b64decode(raw)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'msg': '无法解码图片'})
            return

        # 调用你的检测函数
        result_img, results = _process_image_array(frame)

        # 把标注后的图片转成 base64
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 解析 results
        # 你的 results 结构：(face_counter, fatigue_display, eye_display, ear, area)
        result_list = []
        for item in results:
            face_id = item[0]
            fatigue = item[1]      # "😊 Normal" / "😑 Tired" / "😴 Drowsy"
            eye_state = item[2]    # "👁️ Open" / "👀 Half-closed" / "🚫 Closed"
            ear = item[3]          # EAR 值或 None
            area = item[4]

            result_list.append({
                'id': face_id,
                'fatigue': fatigue,
                'eye': eye_state,
                'ear': f"{ear:.3f}" if ear is not None else "N/A",
                'area': area
            })

        emit('result', {
            'image': f'data:image/jpeg;base64,{img_base64}',
            'results': result_list
        })

    except Exception as e:
        print(f"❌ 处理出错: {e}")
        emit('error', {'msg': str(e)})

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ 客户端已断开')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)