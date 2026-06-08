# Riser cable — PySide6

Phiên bản chuyển từ CustomTkinter sang **PySide6** + làm lại giao diện theo phong cách
industrial dashboard.

## Cấu trúc

```
pyside6/
├── main.py             # entry point — QApplication + load styles + login → main
├── login_window.py     # QDialog đăng nhập (form + scanner status + log)
├── main_window.py      # QMainWindow chính (topbar + image + stats + log + statusbar)
├── scanner.py          # BadgeScanner — QObject chạy trong QThread, phát Signal
├── plc_worker.py       # PLCWorker (base) + SimulatedPLCWorker + PanasonicPLCWorker
├── panasonic.py        # MEWTOCOL driver (giữ nguyên — không phụ thuộc GUI)
├── employees.py        # Danh bạ nhân viên local
├── config.py           # Tham số COM port, baudrate, API URL...
├── INI.py              # Helper đọc/ghi .ini (giữ nguyên)
├── styles.qss          # QSS stylesheet — dark industrial theme
└── requirements.txt
```

## Chạy

```bash
pip install -r requirements.txt
cd pyside6
python main.py
```

## Khác biệt so với bản CustomTkinter

| Bản cũ (CustomTkinter)              | Bản mới (PySide6)                      |
| ---                                 | ---                                    |
| `threading.Thread` + `queue.Queue`  | `QThread` + `Signal`                   |
| GUI drain queue qua `after(80)`     | Signal tự queue sang main thread       |
| 2 file `ctk.CTk` (app riêng cho login & main) | `QApplication` duy nhất, `QDialog` → `QMainWindow` |
| Tô màu inline trong code            | Tách `styles.qss` (QSS)                |
| `CTkImage` cho ảnh                  | Custom `ImageLabel` (QPainter, vẽ placeholder + overlay verdict) |

## Tweak PLC thật

Trong `config.py`:

```python
PLC_SIMULATED = False        # bật chế độ đọc Panasonic thật
PLC_PORT      = "COM1"
PLC_BAUDRATE  = 9600
PLC_POLL_HZ   = 5
```

`PanasonicPLCWorker._poll()` đang là scaffold (đọc 4 word từ `D17800`). Sửa lại
theo logic AOI thật — gọi `self.result.emit({...})` mỗi khi có sản phẩm mới,
`self.image.emit(pil_or_qimage)` khi có ảnh mới.

## Tweak Scanner

Tương tự, `config.py` set `SCANNER_PORT`, `API_TOKEN_URL`,
`API_EMPLOYEE_URL_PREFIX`. Bỏ trống API URL → fallback dùng `employees.py`.
