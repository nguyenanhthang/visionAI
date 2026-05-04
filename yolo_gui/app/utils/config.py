"""
Cấu hình mặc định cho YOLO GUI
Quản lý load/save config và giá trị mặc định theo task
"""
import json
from pathlib import Path

# Tham số huấn luyện mặc định theo từng task
_DEFAULT_CONFIGS = {
    "Classification": {
        "task": "classify",
        "epochs": 100,
        "batch": 16,
        "imgsz": 224,
        "lr0": 0.01,
        "optimizer": "SGD",
        "workers": 8,
        "patience": 50,
        "weight_decay": 0.0005,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "save_period": -1,
        "resume": False,
    },
    "Object Detection": {
        "task": "detect",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "optimizer": "SGD",
        "workers": 8,
        "patience": 50,
        "weight_decay": 0.0005,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "save_period": -1,
        "resume": False,
    },
    "OBB Detection": {
        "task": "obb",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "optimizer": "SGD",
        "workers": 8,
        "patience": 50,
        "weight_decay": 0.0005,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "save_period": -1,
        "resume": False,
    },
    "Segmentation": {
        "task": "segment",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "optimizer": "SGD",
        "workers": 8,
        "patience": 50,
        "weight_decay": 0.0005,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "save_period": -1,
        "resume": False,
    },
}


class AppConfig:
    """Quản lý cấu hình ứng dụng."""

    def __init__(self):
        self.config = {}

    def get_default(self, task: str) -> dict:
        return get_default_config(task)

    def load(self, path: str) -> dict:
        return load_config(path)

    def save(self, config: dict, path: str) -> bool:
        return save_config(config, path)


def get_default_config(task: str) -> dict:
    """Lấy cấu hình mặc định cho task."""
    return _DEFAULT_CONFIGS.get(task, _DEFAULT_CONFIGS["Object Detection"]).copy()


def load_config(path: str) -> dict:
    """Đọc cấu hình từ file JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"File cấu hình không hợp lệ: {e}")


def save_config(config: dict, path: str) -> bool:
    """Lưu cấu hình ra file JSON."""
    try:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Lỗi lưu cấu hình: {e}")
        return False


def validate_config(config: dict) -> bool:
    """Kiểm tra tính hợp lệ của cấu hình."""
    required_keys = ["task", "epochs", "batch", "imgsz", "lr0"]
    for key in required_keys:
        if key not in config:
            return False
    if config.get("epochs", 0) <= 0:
        return False
    if config.get("batch", 0) <= 0:
        return False
    if config.get("imgsz", 0) <= 0:
        return False
    if not (0 < config.get("lr0", 0) <= 1):
        return False
    return True
