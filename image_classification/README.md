# Image Classification với Keras

Một project **train mô hình phân loại ảnh (image classification)** hoàn chỉnh,
gọn gàng và dễ mở rộng, viết bằng **Keras 3 / TensorFlow**. Hỗ trợ cả huấn
luyện CNN từ đầu lẫn **transfer learning** (MobileNetV2, ResNet50, EfficientNet)
kèm giai đoạn **fine-tuning**.

Project đi kèm một bộ dữ liệu tổng hợp (synthetic) nên bạn có thể chạy thử
toàn bộ quy trình `train → evaluate → predict` ngay lập tức mà không cần tải
dataset từ đâu cả.

## Tính năng chính

- 🧱 **Nhiều kiến trúc**: `simple_cnn` (train từ đầu) hoặc backbone pretrained
  (`mobilenetv2`, `resnet50`, `efficientnetb0`).
- 🔁 **Transfer learning + fine-tuning** 2 giai đoạn tự động.
- 🖼️ **Data augmentation** (lật, xoay, zoom, tương phản) nằm ngay trong model,
  nên tự động tắt khi inference.
- ⚙️ **Cấu hình bằng YAML** và có thể override nhanh qua dòng lệnh.
- 📦 **Tiền xử lý nằm trong model** → file `.keras` xuất ra dùng được trực tiếp
  trên ảnh thô, không cần nhớ cách chuẩn hoá.
- 📊 **Đánh giá đầy đủ**: classification report + confusion matrix + biểu đồ
  loss/accuracy.
- 💾 **Callbacks**: lưu checkpoint tốt nhất, early stopping, giảm learning rate,
  TensorBoard, CSV log.

## Cấu trúc thư mục

```
image_classification/
├── README.md
├── requirements.txt
├── Makefile
├── configs/
│   └── default.yaml          # toàn bộ hyperparameter
├── imgcls/                   # package chính (logic tái sử dụng)
│   ├── config.py             # đọc/ghi cấu hình (dataclass + YAML)
│   ├── data.py               # pipeline tf.data từ thư mục ảnh
│   ├── models.py             # các kiến trúc + compile + fine-tune
│   ├── callbacks.py          # khai báo Keras callbacks
│   └── utils.py              # seed, IO, vẽ biểu đồ
├── tools/
│   └── generate_sample_data.py   # sinh dataset tổng hợp để test
├── train.py                  # entrypoint huấn luyện
├── evaluate.py               # entrypoint đánh giá
└── predict.py                # entrypoint dự đoán
```

## Cài đặt

Yêu cầu Python 3.9+.

```bash
cd image_classification
python -m venv .venv && source .venv/bin/activate   # tùy chọn
pip install -r requirements.txt
```

> Máy chỉ có CPU? Có thể cài `tensorflow-cpu` thay cho `tensorflow` cho nhẹ.

## Bắt đầu nhanh (chạy thử ngay)

```bash
# 1) Sinh bộ dữ liệu mẫu (3 lớp: circle / square / triangle)
python tools/generate_sample_data.py --out data/sample --per-class 150

# 2) Huấn luyện (mặc định dùng configs/default.yaml)
python train.py --config configs/default.yaml

# 3) Đánh giá run mới nhất
python evaluate.py --run-dir outputs/<tên_run>

# 4) Dự đoán trên ảnh hoặc cả thư mục
python predict.py --run-dir outputs/<tên_run> --input data/sample/circle
```

Hoặc dùng `Makefile` cho gọn:

```bash
make data      # sinh dữ liệu mẫu
make train     # huấn luyện
make evaluate  # đánh giá run gần nhất
```

## Dùng dataset của riêng bạn

Sắp xếp ảnh theo kiểu **mỗi lớp một thư mục con**:

```
my_dataset/
├── cat/      img001.jpg  img002.jpg ...
├── dog/      ...
└── bird/     ...
```

Rồi trỏ tới nó. Có 2 cách:

**Cách 1 — một thư mục, tự tách train/val:**

```bash
python train.py --dataset-dir my_dataset --backbone resnet50 --epochs 30
```

**Cách 2 — đã chia sẵn train/val/test:**

```yaml
# trong file config
train_dir: my_dataset/train
val_dir:   my_dataset/val
test_dir:  my_dataset/test
dataset_dir: null
```

## Cấu hình

Toàn bộ tham số nằm trong `configs/default.yaml`. Mỗi khóa tương ứng 1:1 với một
trường trong `imgcls/config.py`, và phần lớn có thể override trực tiếp trên dòng
lệnh. Một số tham số quan trọng:

| Khóa | Ý nghĩa |
|------|---------|
| `backbone` | `simple_cnn`, `mobilenetv2`, `resnet50`, `efficientnetb0` |
| `image_size` | kích thước ảnh `[cao, rộng]` |
| `batch_size` | kích thước batch |
| `weights` | `imagenet` (transfer) hoặc `null` (train từ đầu) |
| `epochs` | số epoch giai đoạn 1 (train phần đầu phân loại) |
| `fine_tune` | có chạy giai đoạn fine-tuning hay không |
| `fine_tune_at` | tỉ lệ số layer của backbone bị **đóng băng** khi fine-tune |
| `augment` | bật/tắt data augmentation |

Ví dụ override nhanh:

```bash
python train.py --backbone efficientnetb0 --image-size 224 224 \
                --batch-size 16 --epochs 25 --no-fine-tune
```

## Kết quả huấn luyện sinh ra

Mỗi lần chạy tạo một thư mục `outputs/<run_name>_<timestamp>/` gồm:

```
config.yaml          # snapshot cấu hình đã dùng
class_names.json     # danh sách tên lớp (đúng thứ tự)
best_model.keras     # checkpoint tốt nhất (theo val_accuracy)
final_model.keras    # model ở cuối quá trình train
history.json/.csv    # lịch sử loss/accuracy
training_curves.png  # biểu đồ loss & accuracy
tensorboard/         # log để mở bằng TensorBoard
eval/                # report + confusion matrix (sau khi chạy evaluate.py)
```

Xem TensorBoard:

```bash
tensorboard --logdir outputs/<tên_run>/tensorboard
```

## Cách hoạt động (tóm tắt)

Mỗi model là một `keras.Model` khép kín, nhận ảnh RGB thô `[0, 255]`:

```
inputs → [augmentation] → preprocessing → backbone → head → softmax
```

- **Augmentation** nằm trong model nên chỉ chạy lúc train, tự tắt khi predict.
- **Preprocessing** (chuẩn hoá theo từng backbone) cũng nằm trong model, nên file
  `.keras` xuất ra dùng được ngay trên ảnh thô.
- **Transfer learning**: giai đoạn 1 đóng băng backbone, chỉ train phần head;
  giai đoạn 2 (tùy chọn) mở băng phần trên của backbone và train với learning
  rate rất nhỏ. BatchNorm được giữ đóng băng khi fine-tune để ổn định.

## Mẹo

- Bắt đầu với `mobilenetv2` (nhẹ, nhanh) để dò hyperparameter, rồi đổi sang
  `resnet50`/`efficientnetb0` nếu cần độ chính xác cao hơn.
- Dataset nhỏ → bật `augment` và dùng transfer learning thay vì `simple_cnn`.
- Nếu val accuracy không cải thiện, thử giảm `learning_rate`, tăng `dropout`,
  hoặc tăng cường augmentation.

## License

MIT.
