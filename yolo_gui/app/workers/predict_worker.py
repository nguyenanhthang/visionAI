"""
Worker thread cho quá trình dự đoán YOLO
Xử lý ảnh đơn, thư mục, video và webcam
"""
import traceback

from PySide6.QtCore import QThread, Signal


class PredictWorker(QThread):
    """
    Worker thread thực hiện dự đoán YOLO.
    Hỗ trợ ảnh đơn, thư mục (từng ảnh), video và webcam.
    """

    # Signals
    result = Signal(object, list)   # (annotated_array, detections_list)
    frame = Signal(object)          # numpy array - cho video/webcam
    finished = Signal()
    error = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params.copy()
        self._stop_requested = False

    def stop(self):
        """Yêu cầu dừng dự đoán."""
        self._stop_requested = True

    def run(self):
        """Chạy dự đoán trong thread riêng."""
        self._stop_requested = False

        try:
            from ultralytics import YOLO
        except ImportError:
            self.error.emit(
                "Không tìm thấy thư viện ultralytics.\n"
                "Cài đặt: pip install ultralytics"
            )
            return

        try:
            model_path = self._params.get("model", "")
            if not model_path:
                self.error.emit("Chưa chọn model weights!")
                return

            model = YOLO(model_path)

            # Tham số inference
            predict_kwargs = self._build_predict_kwargs()

            source_type = self._params.get("source_type", "image")

            if source_type == "image":
                self._predict_image(model, predict_kwargs)
            elif source_type == "single_file":
                # Chế độ mới: predict 1 file cụ thể (cho folder next/prev)
                self._predict_single_file(model, predict_kwargs)
            elif source_type == "video":
                self._predict_video(model, predict_kwargs)
            elif source_type == "webcam":
                self._predict_webcam(model, predict_kwargs)
            else:
                self.error.emit(f"Loại nguồn không hợp lệ: {source_type}")
                return

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"❌ Lỗi dự đoán:\n{e}\n\n{tb}")
        finally:
            self.finished.emit()

    def _build_predict_kwargs(self) -> dict:
        """Xây dựng dict tham số cho model.predict()."""
        kwargs = {
            "conf":         self._params.get("conf", 0.25),
            "iou":          self._params.get("iou", 0.7),
            "imgsz":        self._params.get("imgsz", 640),
            "max_det":      self._params.get("max_det", 300),
            "half":         self._params.get("half", False),
            "agnostic_nms": self._params.get("agnostic_nms", False),
            "show_labels":  self._params.get("show_labels", True),
            "show_conf":    self._params.get("show_conf", True),
            "line_width":   self._params.get("line_width", 2),
            "verbose":      False,
        }
        device = self._params.get("device", "")
        if device and device != "auto":
            if ":" in str(device):
                device = str(device).split(":")[0].strip()
            kwargs["device"] = device
        return kwargs

    # ------------------------------------------------------------------
    # Predict methods
    # ------------------------------------------------------------------

    def _predict_image(self, model, kwargs: dict):
        """Dự đoán ảnh đơn."""
        source = self._params.get("source", "")
        if not source:
            self.error.emit("Chưa chọn file ảnh!")
            return
        results = model.predict(source, **kwargs)
        if results:
            r = results[0]
            self.result.emit(r.plot(), self._format_detections(r))

    def _predict_single_file(self, model, kwargs: dict):
        """Dự đoán 1 file ảnh cụ thể (dùng cho folder mode next/prev)."""
        source = self._params.get("source", "")
        if not source:
            self.error.emit("Chưa chọn file ảnh!")
            return
        results = model.predict(source, **kwargs)
        if results:
            r = results[0]
            self.result.emit(r.plot(), self._format_detections(r))

    def _predict_video(self, model, kwargs: dict):
        """Dự đoán từng frame trong video."""
        import cv2
        source = self._params.get("source", "")
        if not source:
            self.error.emit("Chưa chọn file video!")
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.error.emit(f"Không mở được video: {source}")
            return

        last_annotated = None
        last_detections = []

        try:
            while not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, **kwargs)
                if results:
                    annotated = results[0].plot()
                    last_annotated = annotated
                    last_detections = self._format_detections(results[0])
                    self.frame.emit(annotated)
        finally:
            cap.release()

        if last_annotated is not None:
            self.result.emit(last_annotated, last_detections)

    def _predict_webcam(self, model, kwargs: dict):
        """Dự đoán từ webcam."""
        import cv2
        cam_id = self._params.get("cam_id", 0)
        cap = cv2.VideoCapture(int(cam_id))
        if not cap.isOpened():
            self.error.emit(f"Không mở được webcam ID: {cam_id}")
            return

        last_annotated = None
        last_detections = []

        try:
            while not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, **kwargs)
                if results:
                    annotated = results[0].plot()
                    last_annotated = annotated
                    last_detections = self._format_detections(results[0])
                    self.frame.emit(annotated)
        finally:
            cap.release()

        if last_annotated is not None:
            self.result.emit(last_annotated, last_detections)

    # ------------------------------------------------------------------
    # Detection formatting
    # ------------------------------------------------------------------

    def _format_detections(self, result) -> list:
        """Chuyển đổi kết quả YOLO thành list dict."""
        detections = []
        try:
            names = result.names

            # Classification
            if result.probs is not None:
                probs = result.probs
                top5_idx = probs.top5
                top5_conf = (probs.top5conf.tolist()
                             if hasattr(probs.top5conf, "tolist")
                             else list(probs.top5conf))
                for rank, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
                    detections.append({
                        "id": rank + 1,
                        "class": names.get(int(idx), str(idx)),
                        "confidence": round(float(conf), 4),
                        "bbox": "N/A",
                        "area": "N/A",
                    })
                return detections

            # Detection / Segmentation / OBB
            boxes = result.boxes if result.boxes is not None else []
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [round(v, 1) for v in xyxy]
                area = round((x2 - x1) * (y2 - y1), 1)
                detections.append({
                    "id": i + 1,
                    "class": names.get(cls_id, str(cls_id)),
                    "confidence": round(conf, 4),
                    "bbox": f"[{x1},{y1},{x2},{y2}]",
                    "area": area,
                })
        except Exception as e:
            print(f"Lỗi format detections: {e}")
        return detections