# HALCON Vision Studio

GUI PySide6 wrap quanh MVTec HALCON cho các tác vụ machine vision phổ biến.

## Tính năng

- **Image Acquisition** — `open_framegrabber` / `grab_image` (HALCON) hoặc `cv2.VideoCapture` fallback. Hỗ trợ Connect / Live (FPS chỉnh được) / Snapshot / Disconnect.
- **Image Viewer** với zoom (chuột giữa) / pan (drag) / fit / hover toạ độ + giá trị pixel.
- **Blob Analysis** — `threshold` + `connection` + `select_shape` (lọc theo area).
- **Edge Detection (sub-pixel)** — `edges_sub_pix` (canny / sobel / deriche / lanser).
- **Shape-based Matching** — `create_shape_model` + `find_shape_model` (xoay).
  - Lấy template từ file, hoặc **vẽ ROI trực tiếp trên ảnh** (giống `draw_rectangle1` + `crop_rectangle1`), kèm preview & save.
- **Measure 1D** — `gen_measure_rectangle2` + `measure_pairs` (vẽ segment trên ảnh).
- Bảng kết quả + console log + lưu ảnh kết quả.

## Workflow điển hình (production)

1. Mở dock **Acquisition**, chọn interface (DirectShow / GigEVision / OpenCV…) → **Connect**.
2. Bật **Live** để xem stream, hoặc bấm **Snapshot** để chụp 1 frame.
3. Sang tab **Match**, bấm **✎ Pick từ ảnh (ROI)** → kéo chuột chọn vùng mẫu → template được crop tự động.
4. Bấm **▶ Run Match** trên frame mới (hoặc lặp lại snapshot → match cho từng sản phẩm).

## Cài đặt

```bash
pip install PySide6 opencv-python numpy
# Optional: cài MVTec HALCON Python binding (cần licence)
pip install mvtec-halcon  # hoặc theo hướng dẫn của MVTec
```

Nếu thư viện `halcon` không có trong môi trường, app vẫn chạy được — các operator
sẽ fallback sang OpenCV/numpy (status bar sẽ ghi rõ engine đang dùng).

## Chạy

```bash
cd halcon_app
python main.py
```

## Cấu trúc

```
halcon_app/
├── main.py
└── app/
    ├── main_window.py
    ├── styles.py
    ├── operators/
    │   └── halcon_engine.py     # HALCON wrapper + OpenCV fallback
    └── widgets/
        ├── image_canvas.py      # QGraphicsView zoom/pan + measure tool
        ├── operator_panel.py    # Tab parameters cho từng operator
        └── results_view.py      # Bảng metrics + console log
```
