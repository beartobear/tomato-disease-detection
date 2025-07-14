from flask import Flask, render_template, request
import os
import cv2
import base64
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cấu hình Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="FQl2mDWElbti8IH5Ajwh"
)
MODEL_ID = "tomato-detection-q87hl/4"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Trường hợp upload từ file
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

        # Trường hợp ảnh từ webcam (base64)
        elif 'cam_image' in request.form:
            filename = f"camera_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data_url = request.form['cam_image']
            encoded = data_url.split(',')[1]
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(encoded))

        else:
            return "Không có ảnh hợp lệ!", 400

        # Gửi ảnh đến Roboflow
        try:
            result = CLIENT.infer(file_path, model_id=MODEL_ID)
        except Exception as e:
            return f"Lỗi gọi API Roboflow: {e}", 500

        # Vẽ bounding box
        image = cv2.imread(file_path)
        for pred in result['predictions']:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            class_name = pred['class']
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Lưu ảnh kết quả
        annotated_filename = "annotated_" + filename
        annotated_path = os.path.join(OUTPUT_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, image)

        return render_template("result.html", result=result, filename=filename, annotated_filename=annotated_filename)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
