# HALCON Vision Studio

Ứng dụng PySide6 lấy cảm hứng từ Cognex VisionPro, wrap quanh các operator của
MVTec HALCON cho machine vision. Có fallback OpenCV/numpy cho mọi tool nên app
chạy được kể cả khi chưa có HALCON binding / licence.

## Sidebar (collapsible — gập/mở từng nhóm)

| Nhóm           | Tools                                                |
| -------------- | ---------------------------------------------------- |
| 📷 Acquisition | Connect / Live (FPS) / Snapshot / Disconnect         |
| 🪄 Pre-process | Gauss / Median / Mean / Sharpen                      |
| 🎯 Locate      | Blob, Edges (sub-pixel), Pattern Match (+ Pick ROI)  |
| 📐 Measure     | Caliper (Measure 1D), Histogram                      |
| 🔢 Identify    | ID Read (QR / Barcode 1D)                            |
| 🎨 Inspect     | Color stats trong ROI (BGR / RGB / HSV)              |

Click vào header section để mở/gập. Menu **View → Expand/Collapse all sections**
hoặc nút trên toolbar để gập tất cả cùng lúc.

## Mapping HALCON ↔ tool

| Tool            | HALCON operator                                              |
| --------------- | ------------------------------------------------------------ |
| Acquisition     | `open_framegrabber` / `grab_image` / `close_framegrabber`    |
| Filter          | `gauss_filter` / `median_image` / `mean_image` / `emphasize` |
| Blob            | `threshold` + `connection` + `select_shape`                  |
| Edges           | `edges_sub_pix`                                              |
| Pattern Match   | `create_shape_model` + `find_shape_model`                    |
| Caliper         | `gen_measure_rectangle2` + `measure_pairs`                   |
| Histogram       | `gray_histo` (fallback `cv2.calcHist`)                       |
| ID Read         | `find_data_code_2d` (fallback `cv2.QRCodeDetector` + `cv2.barcode`) |
| Color           | `intensity` / `mean_n` trong ROI (BGR / HSV)                 |

## Workflow điển hình

1. **Acquisition** → Connect → Live / Snapshot
2. **Pre-process** → áp filter (output ghi đè ảnh nguồn để xếp chuỗi tool)
3. **Locate** → Pick ROI lấy template → Run Match
4. **Measure / Identify / Inspect** → chạy thêm trên cùng frame

## Cài đặt

```bash
pip install PySide6 opencv-python numpy
pip install mvtec-halcon  # optional, cần licence
```

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
    ├── styles.py                    # theme dark teal kiểu VisionPro
    ├── operators/
    │   ├── halcon_engine.py         # 8 operator + HALCON/OpenCV fallback
    │   └── acquisition.py           # Grabber wrapper
    └── widgets/
        ├── collapsible.py           # CollapsibleSection
        ├── operator_panel.py        # OperatorSidebar (accordion)
        ├── acquisition_panel.py
        ├── image_canvas.py          # zoom/pan + ROI + segment tool
        └── results_view.py
```
