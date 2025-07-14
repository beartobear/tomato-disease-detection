from flask import Flask, render_template, request, redirect
import os
import base64
from PIL import Image, ImageDraw
import io
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Roboflow API
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com", # Corrected URL
    api_key="FQl2mDWElbti8IH5Ajwh"
)
MODEL_ID = "tomato-detection-q87hl/4"

def draw_boxes(image_path, predictions):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x0, y0 = x - w / 2, y - h / 2
        x1, y1 = x + w / 2, y + h / 2

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0 - 10), f"{pred['class']} ({pred['confidence']:.2f})", fill="red")

    # Lưu ảnh kết quả ra thư mục outputs
    output_filename = os.path.basename(image_path).replace(".jpg", "_result.jpg")
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    image.save(output_path)
    return output_filename

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Nếu ảnh từ webcam
        if 'cam_image' in request.form and request.form['cam_image']:
            data_url = request.form['cam_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(binary_data)).convert("RGB")

            filename = "webcam.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
        else:
            # Nếu upload từ máy
            file = request.files["image"]
            if file.filename == "":
                return redirect(request.url)
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        # Gọi Roboflow API
        try:
            result = CLIENT.infer(filepath, model_id=MODEL_ID)
        except Exception as e:
            return f"Lỗi gọi API Roboflow: {e}"

        predictions = result.get("predictions", [])
        annotated_filename = draw_boxes(filepath, predictions)

        # Phân loại ra cà chua và vùng bệnh
        tomatoes = [p for p in predictions if p['class'] in ['ripe', 'unripe']]
        diseases = [p for p in predictions if p['class'] not in ['ripe', 'unripe']]

        return render_template(
            "result.html",
            annotated_filename=annotated_filename,
            result=result,
            tomatoes=tomatoes,
            diseases=diseases
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
