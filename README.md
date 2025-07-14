# 🍅 Phát hiện sâu bệnh cà chua bằng YOLOv8 (Flask + Camera)

Ứng dụng web sử dụng **YOLOv8 thông qua Roboflow API** để phát hiện **sâu bệnh trên lá cà chua** từ ảnh người dùng **upload hoặc chụp từ camera**. Kết quả sẽ được **vẽ bounding box lên ảnh**, trực quan và dễ quan sát.

---

## 🎯 Tính năng chính

- ✅ Upload ảnh từ máy
- ✅ Chụp ảnh trực tiếp bằng webcam trình duyệt
- ✅ Gửi ảnh đến mô hình YOLOv8 được triển khai trên Roboflow
- ✅ Hiển thị kết quả nhận diện với khung vùng bệnh (bounding boxes)
- ✅ Giao diện web đơn giản, dễ sử dụng

---

## 🛠 Công nghệ sử dụng

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/) – vẽ bounding box
- [Roboflow Inference SDK](https://github.com/roboflow/inference)
- HTML5, CSS3, JavaScript (Webcam API)

---

## ⚙️ Cài đặt và chạy ứng dụng

### 1. Clone dự án và cài thư viện

```bash
git clone https://github.com/beartobear/tomato-disease-detection.git
cd tomato-disease-detection
pip install -r requirements.txt
```
- bước 1:
python -m venv venv
.\venv\Scripts\activate
- bước 2:
pip install -r requirements.txt
pip install flask inference-sdk werkzeug
- bước 3:
python app.py

