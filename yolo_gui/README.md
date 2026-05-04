# 🚀 YOLO GUI - Giao diện Huấn luyện và Dự đoán YOLO

Ứng dụng GUI đầy đủ tính năng để huấn luyện và chạy dự đoán với các mô hình [Ultralytics YOLO](https://docs.ultralytics.com), được xây dựng bằng **Python** và **PySide6**.

---

## 📋 Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt](#3-cài-đặt)
4. [Hướng dẫn sử dụng](#4-hướng-dẫn-sử-dụng)
5. [Chuẩn bị Dataset](#5-chuẩn-bị-dataset)
6. [Giải thích Siêu tham số](#6-giải-thích-siêu-tham-số)
7. [FAQ & Xử lý sự cố](#7-faq--xử-lý-sự-cố)

---

## 1. Giới thiệu

**YOLO GUI** cung cấp giao diện đồ họa trực quan để:

- 🏋️ **Huấn luyện** mô hình YOLO với đầy đủ tham số (epochs, batch size, learning rate, augmentation...)
- 🔍 **Dự đoán** trên ảnh đơn, thư mục ảnh, video và webcam
- 📊 **Theo dõi** tiến trình huấn luyện với biểu đồ loss/mAP realtime
- 📋 **Xem log** huấn luyện màu sắc theo mức độ
- 📤 **Xuất kết quả** ra CSV/JSON

### Hỗ trợ 4 task YOLO:
| Task | Mô tả | Model ví dụ |
|------|-------|-------------|
| Classification | Phân loại ảnh | `yolov8n-cls.pt` |
| Object Detection | Phát hiện đối tượng | `yolov8n.pt` |
| OBB Detection | Phát hiện với bbox xoay | `yolov8n-obb.pt` |
| Segmentation | Phân đoạn instance | `yolov8n-seg.pt` |

---

## 2. Yêu cầu hệ thống

### Phần cứng
- **CPU**: Intel Core i5 / AMD Ryzen 5 trở lên
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **GPU**: NVIDIA GPU với CUDA (khuyến nghị cho training) - Tùy chọn
- **Ổ cứng**: 10GB trống

### Phần mềm
- **Python**: 3.8 - 3.11
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **CUDA**: 11.8+ (nếu dùng GPU)

---

## 3. Cài đặt

### Bước 1: Clone hoặc tải về

```bash
cd train/yolo_gui
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Cài đặt PyTorch với CUDA (nếu có GPU)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Bước 5: Khởi chạy ứng dụng

```bash
python main.py
```

---

## 4. Hướng dẫn sử dụng

### 4.1 Tab Train (Huấn luyện) 🚀

**Bước 1 - Chọn Task & Model:**
- Chọn loại task từ dropdown (Classification, Object Detection, OBB, Segmentation)
- Chọn model từ danh sách hoặc dùng nút Browse để chọn file `.pt`
- Model sẽ tự động tải về lần đầu nếu chưa có

**Bước 2 - Chọn Dataset:**
- Nhấn nút `Browse` để chọn file `.yaml` cấu hình dataset
- Kiểm tra thông tin dataset hiển thị bên dưới

**Bước 3 - Cấu hình siêu tham số:**
- Điều chỉnh Epochs, Batch Size, Image Size, Learning Rate
- Chọn Optimizer phù hợp (SGD cho stable, AdamW cho fast)
- Cấu hình Augmentation nếu cần

**Bước 4 - Chọn thiết bị:**
- Ứng dụng tự động phát hiện GPU
- Chọn `auto` để tự động, hoặc chọn GPU/CPU cụ thể

**Bước 5 - Đặt Output:**
- Nhập tên project và experiment
- Chọn thư mục lưu kết quả

**Bước 6 - Huấn luyện:**
- Nhấn `🚀 Bắt đầu Huấn luyện`
- Theo dõi log và biểu đồ metrics ở panel phải
- Nhấn `⏹️ Dừng` để dừng sớm

**Lưu/Tải Config:**
- Dùng `💾 Lưu Config` để lưu cài đặt ra file JSON
- Dùng `📂 Tải Config` để tải lại cài đặt đã lưu

---

### 4.2 Tab Predict (Dự đoán) 🔍

**Bước 1 - Chọn Model:**
- Nhấn Browse để chọn file model đã huấn luyện (`.pt`)
- Ứng dụng hiển thị kích thước file

**Bước 2 - Chọn Nguồn đầu vào:**

| Loại | Mô tả |
|------|-------|
| Ảnh đơn | Một file ảnh (.jpg, .png, ...) |
| Thư mục | Toàn bộ ảnh trong thư mục |
| Video | File video (.mp4, .avi, ...) |
| Webcam | Camera realtime (chọn Camera ID) |

**Bước 3 - Điều chỉnh tham số Inference:**
- **Confidence**: Ngưỡng độ tin cậy (0.25 mặc định)
- **IoU**: Ngưỡng NMS (0.7 mặc định)
- **Max Detections**: Số lượng phát hiện tối đa

**Bước 4 - Chạy dự đoán:**
- Nhấn `🔍 Chạy Dự đoán`
- Xem kết quả ở panel giữa (gốc) và phải (kết quả)
- Bảng bên dưới hiển thị chi tiết từng detection

**Xuất kết quả:**
- `💾 Lưu Kết quả`: Lưu ảnh kết quả
- `📤 Export CSV`: Xuất danh sách detections ra CSV
- `📤 Export JSON`: Xuất ra JSON

---

## 5. Chuẩn bị Dataset

### 5.1 Object Detection

Cấu trúc thư mục:
```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       └── img003.jpg
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── img002.txt
    └── val/
        └── img003.txt
```

File label (`.txt`):
```
# class_id x_center y_center width height (tất cả normalized 0-1)
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

File YAML (tham khảo `configs/detection.yaml`):
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 2
names: {0: cat, 1: dog}
```

---

### 5.2 Classification

Cấu trúc thư mục (mỗi subfolder = 1 class):
```
dataset/
├── train/
│   ├── cat/
│   │   ├── cat001.jpg
│   │   └── cat002.jpg
│   └── dog/
│       └── dog001.jpg
└── val/
    ├── cat/
    └── dog/
```

---

### 5.3 Segmentation

Giống Detection nhưng file label có thêm polygon points:
```
# class_id x1 y1 x2 y2 x3 y3 ... (polygon normalized)
0 0.1 0.2 0.3 0.2 0.3 0.5 0.1 0.5
```

---

### 5.4 OBB Detection (DOTA format)

File label với 8 điểm góc:
```
# class_id x1 y1 x2 y2 x3 y3 x4 y4 (4 góc bbox, normalized)
0 0.1 0.2 0.3 0.1 0.4 0.3 0.2 0.4
```

---

## 6. Giải thích Siêu tham số

| Tham số | Mô tả | Gợi ý |
|---------|-------|-------|
| `epochs` | Số vòng lặp huấn luyện | 100-300 cho dataset nhỏ |
| `batch` | Kích thước batch | 16-32 (tùy VRAM) |
| `imgsz` | Kích thước ảnh input | 640 cho detect, 224 cho classify |
| `lr0` | Learning rate ban đầu | 0.01 (SGD), 0.001 (Adam) |
| `optimizer` | Thuật toán tối ưu | SGD (stable), AdamW (fast) |
| `patience` | Early stopping | 50 (dừng nếu không cải thiện) |
| `weight_decay` | Regularization L2 | 0.0005 (mặc định) |
| `mosaic` | Augmentation ghép 4 ảnh | 1.0 (bật), 0.0 (tắt) |
| `mixup` | Trộn 2 ảnh | 0.0-0.3 |
| `fliplr` | Xác suất lật ngang | 0.5 (mặc định) |
| `degrees` | Góc xoay tối đa | 0 (không xoay), 10-15 (xoay nhẹ) |
| `save_period` | Lưu checkpoint mỗi N epoch | -1 (chỉ best/last) |

### Optimizer

| Optimizer | Đặc điểm | Khi nào dùng |
|-----------|----------|--------------|
| SGD | Ổn định, cần momentum | Training lâu dài |
| Adam | Hội tụ nhanh | Dataset nhỏ |
| AdamW | Adam + weight decay | Thường dùng nhất |
| auto | Tự động chọn | Mặc định |

---

## 7. FAQ & Xử lý sự cố

### ❓ CUDA out of memory

**Giải pháp:**
- Giảm `batch size` (từ 16 xuống 8 hoặc 4)
- Giảm `imgsz` (từ 640 xuống 416 hoặc 320)
- Dùng model nhỏ hơn (n thay vì m hoặc l)
- Bật `half precision` trong tab Predict

---

### ❓ Model không tải được

**Giải pháp:**
- Kiểm tra kết nối internet (lần đầu cần tải model)
- Đảm bảo đủ dung lượng ổ cứng
- Thử chạy: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`

---

### ❓ Không tìm thấy GPU

**Giải pháp:**
- Cài đặt PyTorch với CUDA đúng phiên bản
- Kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`
- Cài lại CUDA driver

---

### ❓ Training rất chậm

**Giải pháp:**
- Tăng `workers` (nếu có nhiều CPU core)
- Dùng GPU thay vì CPU
- Tăng batch size (nếu đủ RAM/VRAM)
- Dùng model nhỏ hơn (yolov8n thay vì yolov8x)

---

### ❓ mAP thấp

**Giải pháp:**
- Kiểm tra chất lượng nhãn (labels)
- Tăng số epochs
- Thử model lớn hơn
- Điều chỉnh learning rate (thử 0.001 hoặc 0.0001)
- Thêm augmentation (mosaic, mixup)
- Thu thập thêm dữ liệu

---

### ❓ Lỗi "No module named 'ultralytics'"

```bash
pip install ultralytics
```

---

### ❓ Lỗi PySide6 trên Linux

```bash
sudo apt-get install libglib2.0-0 libxcb-xinerama0 libxcb-cursor0
pip install PySide6
```

---

## 📞 Liên hệ & Đóng góp

- 🐛 Báo lỗi: Mở Issue trên GitHub
- 📚 Tài liệu YOLO: https://docs.ultralytics.com
- 💬 Cộng đồng: https://community.ultralytics.com
