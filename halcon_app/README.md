# HALCON Vision Studio

GUI PySide6 wrap quanh MVTec HALCON cho các tác vụ machine vision phổ biến.

## Tính năng

- **Image Viewer** với zoom (chuột giữa) / pan (drag) / fit / hover toạ độ + giá trị pixel.
- **Blob Analysis** — `threshold` + `connection` + `select_shape` (lọc theo area).
- **Edge Detection (sub-pixel)** — `edges_sub_pix` (canny / sobel / deriche / lanser).
- **Shape-based Matching** — `create_shape_model` + `find_shape_model` (xoay).
- **Measure 1D** — `gen_measure_rectangle2` + `measure_pairs` (vẽ segment trên ảnh).
- Bảng kết quả + console log + lưu ảnh kết quả.

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
