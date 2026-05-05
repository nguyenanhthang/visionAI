# HALCON Vision Studio

Ứng dụng PySide6 lấy cảm hứng từ Cognex VisionPro, wrap quanh các operator của
MVTec HALCON cho machine vision. Có fallback OpenCV/numpy cho mọi tool nên app
chạy được kể cả khi chưa có HALCON binding / licence.

## Sidebar (collapsible — gập/mở từng nhóm)

| Nhóm           | Tools                                                              |
| -------------- | ------------------------------------------------------------------ |
| 📷 Acquisition | Connect / Live (FPS) / Snapshot / Disconnect                       |
| 🪄 Pre-process | Filter (Gauss/Median/Mean/Sharpen) + Morphology (dilate/erode/…)   |
| 🩹 Mask / ROI  | Sinh từ gray range / HSV / ROI vẽ tay; invert/clear/save/load      |
| 🎯 Locate      | Blob, Adaptive Threshold, Edges (sub-pixel), Contours, Pattern Match (+ Pick ROI) |
| 📐 Measure     | Caliper (Measure 1D), Histogram                                    |
| 🔢 Identify    | ID Read (QR / Barcode 1D)                                          |
| 🎨 Inspect     | Color stats trong ROI; Image Diff (golden template)                |

### Mask system
- Sinh mask từ **gray range**, **HSV range** hoặc **ROI vẽ tay** trên ảnh.
- Invert / Clear / Save / Load mask (PNG nhị phân).
- Toggle hiện/ẩn overlay teal trên canvas.
- Khi mask được set, **mọi operator** đều áp `apply_mask` (HALCON: `reduce_domain`).
- Status badge "mask 12.7%" hiển thị tỷ lệ pixel trong mask.

### Layout
- Canvas chiếm ~78% chiều dọc, results panel ~22%; có thể **collapse Results** bằng nút `▾ Hide`.
- Status badges trên header viewer: nguồn ảnh (`idle/file/connect/live/snapshot`) và mask coverage.

Click vào header section để mở/gập. Menu **View → Expand/Collapse all sections**
hoặc nút trên toolbar để gập tất cả cùng lúc.

## Mapping HALCON ↔ tool

| Tool                | HALCON operator                                                       |
| ------------------- | --------------------------------------------------------------------- |
| Acquisition         | `open_framegrabber` / `grab_image` / `close_framegrabber`             |
| Filter              | `gauss_filter` / `median_image` / `mean_image` / `emphasize`          |
| Morphology          | `dilation_circle` / `erosion_circle` / `opening_circle` / `closing_circle` |
| Mask (gray / HSV)   | `threshold` (+ `reduce_domain` áp lên operator)                       |
| Blob                | `threshold` + `connection` + `select_shape`                           |
| Adaptive Threshold  | `dyn_threshold` (fallback `cv2.adaptiveThreshold`)                    |
| Edges               | `edges_sub_pix`                                                       |
| Contours            | `gen_contours_skeleton_xld` / `select_contours_xld`                   |
| Pattern Match       | `create_shape_model` + `find_shape_model`                             |
| Caliper             | `gen_measure_rectangle2` + `measure_pairs`                            |
| Histogram           | `gray_histo` (fallback `cv2.calcHist`)                                |
| ID Read             | `find_data_code_2d` (fallback `cv2.QRCodeDetector` + `cv2.barcode`)   |
| Color               | `intensity` / `mean_n` trong ROI (BGR / HSV)                          |
| Image Diff          | `sub_image` + `threshold` + `connection` (golden compare)             |

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
