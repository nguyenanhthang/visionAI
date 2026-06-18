"""Danh bạ nhân viên - load từ file Python ngoài để không cần build lại app.

Khi đóng gói (PyInstaller):
    File 'employees_data.py' nằm CẠNH file .exe (KHÔNG bundle vào trong).
    IT chỉnh dict EMPLOYEE_DIRECTORY trong file đó, restart app là xong.

Vì sao tên ngoài là 'employees_data.py' chứ không phải 'employees.py'?
    Nếu trùng tên với module này thì khi chạy dev, loader sẽ tự load chính
    nó (loader code không có EMPLOYEE_DIRECTORY) -> data rỗng. Tách tên ra
    để cả dev và prod đều chạy đúng.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional


def _data_dir() -> Path:
    """Thư mục chứa file ngoài.

    - Đóng gói (sys.frozen=True): cạnh .exe (sys.executable.parent)
    - Dev (python main.py): cạnh script .py

    KHÔNG dùng __file__ khi frozen vì PyInstaller extract code vào _MEIPASS
    (temp folder), không phải nơi đặt .exe.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


EXTERNAL_FILE = _data_dir() / "employees_data.py"


_DEFAULT_TEMPLATE = '''"""Danh sách nhân viên - sửa file này không cần build lại app.

Thêm/xóa entry trong dict bên dưới. Lưu lại rồi restart app
(hoặc gọi employees.reload() nếu app có nút Reload).
"""

EMPLOYEE_DIRECTORY = {
    "001": "Nguyễn Văn A",
    "002": "Trần Thị B",
    "003": "Lê Văn C",
}
'''


_cache: Dict[str, str] = {}
_loaded = False


def _ensure_file_exists():
    if EXTERNAL_FILE.exists():
        return
    try:
        EXTERNAL_FILE.write_text(_DEFAULT_TEMPLATE, encoding="utf-8")
    except Exception:
        pass  # ổ chỉ đọc / no permission -> đành dùng cache rỗng


def _load():
    global _cache, _loaded
    _ensure_file_exists()
    _cache = {}
    if EXTERNAL_FILE.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "_employees_external", str(EXTERNAL_FILE)
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                data = getattr(module, "EMPLOYEE_DIRECTORY", {})
                if isinstance(data, dict):
                    _cache = {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass  # syntax lỗi trong file ngoài -> cache rỗng, app vẫn chạy
    _loaded = True


def reload():
    """Nạp lại file ngoài. Gọi sau khi sửa employees_data.py mà không restart."""
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
    """Trả đường dẫn file ngoài - để hiển thị trong UI / log lỗi."""
    return EXTERNAL_FILE
