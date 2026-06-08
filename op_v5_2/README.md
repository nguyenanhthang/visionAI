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
├── plc_worker.py       # PLCWorker (base) + SimulatedPLCWorker + H3U_PLCWorker
├── h3u_h5u.py          # Inovance H3U/H5U Modbus TCP driver (đang dùng)
├── panasonic.py        # MEWTOCOL driver Panasonic (driver thay thế)
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

## Tweak PLC thật (Inovance H3U/H5U — Modbus TCP)

Trong `config.py`:

```python
PLC_SIMULATED        = False           # False → đọc PLC thật qua Modbus TCP
PLC_IP               = '192.168.250.1'
PLC_PORT             = 502             # cổng Modbus TCP (mặc định 502)
PLC_RESULT_ADDR      = 300             # holding register AOI ghi verdict: 1=OK, 2=NG
PLC_SCAN_RESULT_ADDR = 250             # holding register Scanner ghi sau check SFC
PLC_SCAN_CHECK_ADDR  = 500             # holding register PLC bật =1 hỏi "đã quét SN chưa"
PLC_POLL_HZ          = 5
```

`H3U_PLCWorker` poll reg `PLC_RESULT_ADDR`: đọc 1 → `result=PASS`, 2 → `result=FAIL`,
rồi ghi lại 0. Đồng thời poll reg `PLC_SCAN_CHECK_ADDR`: khi PLC bật lên 1 (sườn lên)
mà toggle **Quét SN** đang ON nhưng chưa có SN → log lỗi "CHƯA QUÉT HÀNG".
Giao thức nằm trong `h3u_h5u.py` (modbus-tk, holding register, slave id 1).

Giao diện chính có hàng **ĐIỀU KHIỂN** với 3 toggle: **Quét SN** (OFF → tự ghi
reg 250 = 1, bỏ qua quét tay), **Lưu Excel**, **Lưu ảnh**. Mặc định set trong
`config.py` (`SCAN_ENABLED` / `SAVE_EXCEL` / `SAVE_IMAGE`) hoặc tab *Tính năng*
trong Settings.

## Tweak Scanner

Tương tự, `config.py` set `SCANNER_PORT`, `API_TOKEN_URL`,
`API_EMPLOYEE_URL_PREFIX`. Bỏ trống API URL → fallback dùng `employees.py`.
