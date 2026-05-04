"""
Tiện ích file cho YOLO GUI
Xử lý tìm kiếm file ảnh/video và lưu kết quả
"""
import csv
import json
from pathlib import Path

# Định dạng file ảnh được hỗ trợ
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
# Định dạng file video được hỗ trợ
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]


def get_image_files(directory: str) -> list:
    """Lấy danh sách tất cả file ảnh trong thư mục."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(dir_path.rglob(f"*{ext}"))
        files.extend(dir_path.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def get_video_files(directory: str) -> list:
    """Lấy danh sách tất cả file video trong thư mục."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(dir_path.rglob(f"*{ext}"))
        files.extend(dir_path.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def ensure_dir(path: str) -> Path:
    """Tạo thư mục nếu chưa tồn tại."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results_csv(results: list, path: str) -> bool:
    """Lưu kết quả dự đoán ra file CSV."""
    if not results:
        return False
    try:
        out_path = Path(path)
        ensure_dir(out_path.parent)
        fieldnames = results[0].keys() if results else []
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        return True
    except Exception as e:
        print(f"Lỗi khi lưu CSV: {e}")
        return False


def save_results_json(results: list, path: str) -> bool:
    """Lưu kết quả dự đoán ra file JSON."""
    if not results:
        return False
    try:
        out_path = Path(path)
        ensure_dir(out_path.parent)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        print(f"Lỗi khi lưu JSON: {e}")
        return False
