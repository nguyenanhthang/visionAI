"""
Worker thread cho quá trình huấn luyện YOLO
Chạy trong QThread riêng để không block UI
"""
import io
import sys
import time
import traceback

from PySide6.QtCore import QThread, Signal


class TrainWorker(QThread):
    """
    Worker thread thực hiện huấn luyện YOLO.
    Giao tiếp với UI thông qua signals.
    """

    # Signals
    progress = Signal(int, int, dict)   # (epoch hiện tại, tổng epoch, metrics_dict)
    log = Signal(str)                    # Dòng log
    metrics = Signal(dict)               # Dict metrics sau mỗi epoch
    finished = Signal(str)               # Đường dẫn kết quả khi hoàn thành
    error = Signal(str)                  # Thông báo lỗi

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params.copy()
        self._stop_requested = False
        self._start_time = 0.0
        self._current_epoch = 0
        self._total_epochs = params.get("epochs", 100)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self):
        """Yêu cầu dừng huấn luyện."""
        self._stop_requested = True
        self.log.emit("⚠️ Đã gửi yêu cầu dừng huấn luyện...")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self):
        """Chạy quá trình huấn luyện trong thread riêng."""
        self._start_time = time.time()
        self._stop_requested = False

        # Redirect stdout/stderr để capture log
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        log_buffer = _LogCapture(self.log)
        sys.stdout = log_buffer
        sys.stderr = log_buffer

        try:
            self.log.emit("🚀 Bắt đầu huấn luyện YOLO...")
            self.log.emit(f"📋 Tham số: {self._params}")

            # Import YOLO
            try:
                from ultralytics import YOLO
            except ImportError:
                self.error.emit(
                    "Không tìm thấy thư viện ultralytics.\n"
                    "Cài đặt: pip install ultralytics"
                )
                return

            model_path = self._params.get("model", "yolov8n.pt")
            self.log.emit(f"📦 Tải model: {model_path}")

            model = YOLO(model_path)

            # Đăng ký callbacks
            self._register_callbacks(model)

            # Xây dựng kwargs cho model.train()
            train_kwargs = self._build_train_kwargs()

            self.log.emit("🎯 Bắt đầu training...")
            self.log.emit(f"⚙️ Config: epochs={train_kwargs.get('epochs')}, "
                          f"batch={train_kwargs.get('batch')}, "
                          f"imgsz={train_kwargs.get('imgsz')}")

            # Bắt đầu training
            results = model.train(**train_kwargs)

            if self._stop_requested:
                self.log.emit("⏹️ Huấn luyện đã được dừng theo yêu cầu.")
                self.finished.emit("")
                return

            # Lấy đường dẫn kết quả
            save_dir = ""
            if results is not None:
                if hasattr(results, "save_dir"):
                    save_dir = str(results.save_dir)
                elif hasattr(model, "trainer") and model.trainer:
                    save_dir = str(model.trainer.save_dir)

            elapsed = time.time() - self._start_time
            self.log.emit(
                f"✅ Huấn luyện hoàn thành! "
                f"Thời gian: {_fmt_time(elapsed)}. "
                f"Kết quả: {save_dir}"
            )
            self.finished.emit(save_dir)

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"❌ Lỗi huấn luyện:\n{e}\n\nTraceback:\n{tb}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_train_kwargs(self) -> dict:
        """Xây dựng dict tham số cho model.train()."""
        p = self._params
        kwargs = {
            "data":         str(p.get("data", "")),
            "epochs":       int(p.get("epochs", 100)),
            "batch":        int(p.get("batch", 16)),
            "imgsz":        int(p.get("imgsz", 640)),
            "lr0":          float(p.get("lr0", 0.01)),
            "optimizer":    str(p.get("optimizer", "SGD")),
            "workers":      int(p.get("workers", 8)),
            "patience":     int(p.get("patience", 50)),
            "weight_decay": float(p.get("weight_decay", 0.0005)),
            "hsv_h":        float(p.get("hsv_h", 0.015)),
            "hsv_s":        float(p.get("hsv_s", 0.7)),
            "hsv_v":        float(p.get("hsv_v", 0.4)),
            "degrees":      float(p.get("degrees", 0.0)),
            "translate":    float(p.get("translate", 0.1)),
            "scale":        float(p.get("scale", 0.5)),
            "flipud":       float(p.get("flipud", 0.0)),
            "fliplr":       float(p.get("fliplr", 0.5)),
            "mosaic":       float(p.get("mosaic", 1.0)),
            "mixup":        float(p.get("mixup", 0.0)),
            "save_period":  int(p.get("save_period", -1)),
            "project":      str(p.get("project", "runs")),
            "name":         str(p.get("name", "train")),
            "exist_ok":     True,
            "verbose":      True,
        }

        # Device - xử lý riêng
        device = p.get("device", "")
        if device and device != "auto":
            if ":" in str(device):
                device = str(device).split(":")[0].strip()
            kwargs["device"] = device

        # Resume - CHỈ truyền path hoặc không truyền gì
        # ❌ KHÔNG truyền resume=False (gây conflict)
        if p.get("resume") and p.get("resume_path"):
            kwargs["resume"] = str(p["resume_path"])
        # Nếu không resume thì KHÔNG thêm key "resume" vào kwargs

        # Xóa key có giá trị rỗng
        kwargs = {k: v for k, v in kwargs.items() if v != ""}
        return kwargs

    def _register_callbacks(self, model):
        """Đăng ký callbacks để capture metrics từ YOLO."""
        worker = self

        def on_train_epoch_end(trainer):
            if worker._stop_requested:
                trainer.stop = True
                return
            epoch = trainer.epoch + 1
            total = trainer.epochs
            worker._current_epoch = epoch
            metrics_dict = {"epoch": epoch}
            if hasattr(trainer, "metrics") and trainer.metrics:
                metrics_dict.update(trainer.metrics)
            if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                try:
                    loss_names = getattr(trainer, "loss_names", [])
                    items = trainer.loss_items
                    if hasattr(items, "tolist"):
                        items = items.tolist()
                    for name, val in zip(loss_names, items):
                        metrics_dict[f"train/{name}"] = float(val)
                except Exception:
                    pass
            elapsed = time.time() - worker._start_time
            eta = (elapsed / epoch * (total - epoch)) if epoch > 0 else 0
            speed = f"{elapsed / epoch:.1f}s/epoch" if epoch > 0 else "--"
            worker.progress.emit(epoch, total, {
                "elapsed": elapsed, "eta": eta, "speed": speed,
            })
            worker.metrics.emit(metrics_dict)

        def on_train_start(trainer):
            worker.log.emit(f"🏋️ Bắt đầu training | Device: {trainer.device}")
            worker.log.emit(f"📁 Lưu kết quả tại: {trainer.save_dir}")

        def on_train_end(trainer):
            worker.log.emit("🎉 Training hoàn thành!")
            if hasattr(trainer, "best"):
                worker.log.emit(f"🏆 Model tốt nhất: {trainer.best}")

        try:
            # ✅ FIX: Dùng model.callbacks dict trực tiếp thay vì add_callback
            # add_callback sẽ APPEND vào list → gây lỗi 'list' not callable
            # Cách đúng: append vào list callbacks (Ultralytics sẽ iterate qua list)
            
            # Kiểm tra xem callbacks đã là list chưa
            if isinstance(model.callbacks.get("on_train_epoch_end"), list):
                model.callbacks["on_train_epoch_end"].append(on_train_epoch_end)
            else:
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                
            if isinstance(model.callbacks.get("on_train_start"), list):
                model.callbacks["on_train_start"].append(on_train_start)
            else:
                model.add_callback("on_train_start", on_train_start)
                
            if isinstance(model.callbacks.get("on_train_end"), list):
                model.callbacks["on_train_end"].append(on_train_end)
            else:
                model.add_callback("on_train_end", on_train_end)
                
        except Exception as e:
            self.log.emit(f"⚠️ Không thể đăng ký callback: {e}")


class _LogCapture(io.StringIO):
    """Capture stdout/stderr và emit qua signal."""

    def __init__(self, signal: Signal):
        super().__init__()
        self._signal = signal

    def write(self, text: str):
        if text.strip():
            self._signal.emit(text.rstrip())
        return len(text)

    def flush(self):
        pass


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
