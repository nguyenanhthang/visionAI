# Model OCR offline — chạy OCR Max KHÔNG cần mạng

Tool **OCR Max** dùng **PaddleOCR**. Để chạy **offline**, đặt sẵn model vào
`models/paddle/` (làm 1 lần), app sẽ không bao giờ gọi internet khi Run.

```
models/
└── paddle/
    ├── det/   # model phát hiện vùng chữ (chứa inference.pd*)
    ├── rec/   # model nhận dạng ký tự
    └── cls/   # model xoay góc (tùy bản — có thể không có, vẫn chạy được)
```

App tự dò model theo thứ tự ưu tiên:
1. Param *PaddleOCR models folder* nhập trong node
2. `<app>/models/paddle` (thư mục này)
3. `~/.paddleocr` (nơi PaddleOCR tự lưu sau lần tải đầu)

---

## 1. Cài đặt
```bash
pip install paddlepaddle      # bản CPU (GPU: paddlepaddle-gpu)
pip install paddleocr
```

## 2. Lấy model offline (1 lần trên máy CÓ mạng)
```bash
python tools/fetch_ocr_models.py --lang vi      # tiếng Việt
python tools/fetch_ocr_models.py --lang en      # tiếng Anh
```
Lệnh gọi PaddleOCR tự tải model rồi copy vào `models/paddle/{det,rec,cls}`.
Sau đó **copy nguyên thư mục `models/`** sang máy offline (đặt cạnh app / .exe).

> **Thủ công:** chạy app 1 lần khi CÓ mạng (Run node OCR Max) → model về
> `~/.paddleocr/whl/`. Copy 3 folder con (folder chứa `inference.pdmodel`) vào
> `models/paddle/det`, `/rec`, `/cls`.

## 3. Dùng
Node OCR Max → **Language = `vie`** (hoặc `en`, `japan`…). Có model local → chạy
offline. Thiếu model + không mạng → báo lỗi chỉ rõ thư mục cần đặt.

---

## Đóng gói (PyInstaller)

`VisionPro.spec` đã bundle cả thư mục `models/`. Đặt model vào `models/paddle/`
trước khi build → `.exe` mang theo model, máy đích chạy offline ngay.
