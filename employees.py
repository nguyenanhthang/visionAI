"""Danh bạ nhân viên - đọc từ file CSV ngoài để không cần build lại app.

Khi đóng gói (PyInstaller / cx_Freeze):
    File 'employees.csv' nằm CẠNH file .exe, KHÔNG bundle vào trong.
    Sửa CSV bằng Notepad/Excel rồi gọi reload() (hoặc restart app).

Format CSV:
    # comment (bỏ qua)
    mã_nv,tên_nv
    001,Nguyễn Văn A
    ...

Lần đầu chạy mà thiếu file -> tự tạo template cạnh .exe.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, Optional


# Template mặc định khi file chưa tồn tại
_DEFAULT_EMPLOYEES = {
    "001": "Nguyễn Văn A",
    "002": "Trần Thị B",
    "003": "Lê Văn C",
}


def _data_dir() -> Path:
    """Thư mục chứa file dữ liệu.

    - Khi chạy bằng PyInstaller/cx_Freeze (sys.frozen=True): cạnh file .exe
    - Khi chạy dev (python main.py): cạnh script .py

    KHÔNG dùng __file__ trong frozen mode vì PyInstaller extract vào temp
    folder _MEIPASS - file CSV ngoài sẽ không nằm ở đó.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


EMPLOYEES_FILE = _data_dir() / "employees.csv"

_cache: Dict[str, str] = {}
_loaded = False


def _ensure_file_exists():
    """Lần đầu chạy mà thiếu CSV -> tạo template."""
    if EMPLOYEES_FILE.exists():
        return
    try:
        with EMPLOYEES_FILE.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["# Mã NV", "Tên NV"])
            writer.writerow(["# Sửa file này không cần build lại app", ""])
            for emp_id, name in _DEFAULT_EMPLOYEES.items():
                writer.writerow([emp_id, name])
    except Exception:
        pass  # không tạo được (ví dụ ổ chỉ đọc) -> đành dùng cache rỗng


def _load():
    global _cache, _loaded
    _ensure_file_exists()
    _cache = {}
    if not EMPLOYEES_FILE.exists():
        _loaded = True
        return
    try:
        with EMPLOYEES_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                emp_id = row[0].strip()
                if not emp_id or emp_id.startswith("#"):
                    continue
                name = row[1].strip() if len(row) > 1 else ""
                _cache[emp_id] = name
    except Exception:
        # CSV format lỗi -> cache rỗng, app vẫn chạy
        pass
    _loaded = True


def reload():
    """Nạp lại file CSV. Gọi sau khi sửa file mà không muốn restart."""
    _load()


def lookup(employee_id: str) -> Optional[str]:
    if not _loaded:
        _load()
    return _cache.get(employee_id)


def exists(employee_id: str) -> bool:
    if not _loaded:
        _load()
    return employee_id in _cache


def all_employees() -> Dict[str, str]:
    if not _loaded:
        _load()
    return dict(_cache)


def file_path() -> Path:
    """Trả đường dẫn file CSV - để hiển thị trong UI / log lỗi."""
    return EMPLOYEES_FILE
