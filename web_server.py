# web_server.py
import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import dlib

# 导入核心检测模块（不依赖 tkinter）
from detection_core import _process_image_array, load_model, dlib_predictor, dlib_available
import detection_core

# 手动初始化 dlib（如果 detection_core 里没有自动初始化）
dat_file = 'shape_predictor_68_face_landmarks.dat'
if os.path.exists(dat_file):
    try:
        detection_core.dlib_predictor = dlib.shape_predictor(dat_file)
        detection_core.dlib_available = True
        print("✅ dlib predictor loaded in web_server")
    except Exception as e:
        print(f"❌ Failed to load dlib: {e}")
        detection_core.dlib_available = False
else:
    print(f"❌ {dat_file} not found, EAR disabled")
    detection_core.dlib_available = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

print("⏳ Loading YOLO model...")
load_model()
print("✅ YOLO model loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected')
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
            emit('error', {'msg': 'Cannot decode image'})
            return

        result_img, results = _process_image_array(frame)
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        result_list = []
        for item in results:
            # item: (face_counter, fatigue_display, eye_display, ear, area)
            result_list.append({
                'id': item[0],
                'fatigue': item[1],
                'eye': item[2],
                'ear': f"{item[3]:.3f}" if item[3] is not None else "N/A",
                'area': item[4]
            })

        emit('result', {
            'image': f'data:image/jpeg;base64,{img_base64}',
            'results': result_list
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        emit('error', {'msg': str(e)})

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
