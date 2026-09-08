# web_server.py
import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from detection_core import _process_image_array, load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

model_loaded = False

print("⏳ Web server starting (model will load on first request)...")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected')
    emit('ready', {'status': 'connected'})

@socketio.on('image')
def handle_image(data):
    global model_loaded
    try:
        if not model_loaded:
            print("⏳ Loading YOLO model on first request...")
            load_model()
            model_loaded = True
            print("✅ YOLO model loaded.")

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
