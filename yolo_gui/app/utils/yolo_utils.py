"""
Tiện ích YOLO cho GUI
Cung cấp thông tin model, thiết bị và các hàm phân tích log
"""
import re
from typing import Optional

# Danh sách model theo từng task
MODELS = {
    "Classification": [
        "yolov8n-cls.pt",
        "yolov8s-cls.pt",
        "yolov8m-cls.pt",
        "yolov8l-cls.pt",
        "yolov8x-cls.pt",
        "yolo11n-cls.pt",
        "yolo11s-cls.pt",
        "yolo11m-cls.pt",
    ],
    "Object Detection": [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
    ],
    "OBB Detection": [
        "yolov8n-obb.pt",
        "yolov8s-obb.pt",
        "yolov8m-obb.pt",
        "yolov8l-obb.pt",
        "yolov8x-obb.pt",
    ],
    "Segmentation": [
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
        "yolov8l-seg.pt",
        "yolov8x-seg.pt",
        "yolo11n-seg.pt",
        "yolo11s-seg.pt",
    ],
}

# Mapping task name -> YOLO task key
TASK_MAP = {
    "Classification": "classify",
    "Object Detection": "detect",
    "OBB Detection": "obb",
    "Segmentation": "segment",
}


def get_available_models(task: str) -> list:
    """Lấy danh sách model cho task được chọn."""
    return MODELS.get(task, [])


def get_device_info() -> dict:
    """Lấy thông tin thiết bị GPU/CPU."""
    info = {"device": "cpu", "name": "CPU", "vram": 0, "devices": ["cpu"]}
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            info["device"] = "0"
            info["name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["vram"] = round(vram, 1)
            info["devices"] = [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(count)]
            info["devices"].append("cpu")
        else:
            info["devices"] = ["cpu"]
    except ImportError:
        pass
    except Exception as e:
        print(f"Lỗi lấy thông tin thiết bị: {e}")
    return info


def parse_training_log(line: str) -> Optional[dict]:
    """
    Phân tích dòng log huấn luyện YOLO để lấy metrics.
    Trả về dict metrics hoặc None nếu không có dữ liệu.
    """
    metrics = {}
    # Pattern nhận diện epoch progress: "1/100"
    epoch_match = re.search(r"(\d+)/(\d+)\s+\d+\.\d+[Gg]", line)
    if epoch_match:
        metrics["epoch"] = int(epoch_match.group(1))
        metrics["total_epochs"] = int(epoch_match.group(2))

    # Lấy loss values
    for key in ["box_loss", "cls_loss", "dfl_loss", "seg_loss"]:
        match = re.search(rf"{key}\s+([\d.]+)", line)
        if match:
            metrics[f"train/{key}"] = float(match.group(1))

    # Lấy mAP metrics
    for key in ["mAP50", "mAP50-95", "precision", "recall"]:
        match = re.search(rf"{key}\s+([\d.]+)", line, re.IGNORECASE)
        if match:
            metrics[f"metrics/{key}"] = float(match.group(1))

    return metrics if metrics else None


def get_results_summary(results) -> dict:
    """Lấy tóm tắt kết quả huấn luyện từ đối tượng results YOLO."""
    summary = {}
    try:
        if hasattr(results, "results_dict"):
            summary = results.results_dict
        elif hasattr(results, "maps"):
            summary["mAP50-95"] = float(results.maps.mean())
        if hasattr(results, "save_dir"):
            summary["save_dir"] = str(results.save_dir)
    except Exception as e:
        summary["error"] = str(e)
    return summary
