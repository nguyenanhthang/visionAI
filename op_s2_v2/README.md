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
├── plc_worker.py       # PLCWorker (base) + SimulatedPLCWorker + CP2E_PLCWorker
├── cp2e.py             # Omron CP2E FINS/TCP driver (đang dùng)
├── h3u_h5u.py          # Inovance H3U/H5U Modbus TCP driver (driver thay thế)
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

## Tweak PLC thật (Omron CP2E — FINS/TCP)

Trong `config.py`:

```python
PLC_SIMULATED        = False           # False → đọc CP2E thật qua FINS/TCP
PLC_IP               = '192.168.250.1'
PLC_PORT             = 9600            # cổng FINS/TCP (Omron mặc định 9600)
PLC_RESULT_ADDR      = 300             # DM word AOI ghi verdict: 1=OK, 2=NG (D300)
PLC_SCAN_RESULT_ADDR = 250             # DM word Scanner ghi sau check SFC  (D250)
PLC_POLL_HZ          = 5
```

`CP2E_PLCWorker` poll `D<PLC_RESULT_ADDR>`: đọc 1 → `result=PASS`, 2 → `result=FAIL`,
rồi ghi lại 0. Giao thức nằm trong `cp2e.py` (vùng Data Memory, area code `0x82`).
Đổi PLC khác chỉ cần viết driver cùng API module-level rồi đổi import trong
`plc_worker.py` + `scanner.py` (xem `h3u_h5u.py` / `panasonic.py` làm mẫu).

## Tweak Scanner

Tương tự, `config.py` set `SCANNER_PORT`, `API_TOKEN_URL`,
`API_EMPLOYEE_URL_PREFIX`. Bỏ trống API URL → fallback dùng `employees.py`.

## Fix "app đơ sau ~2 giờ chạy" (2026-06)

Nguyên nhân chính: handshake verdict D300 kiểu **so sánh với giá trị poll
trước** + ghi ack 0 SAU khi emit. Chạy vài giờ = vài chục nghìn giao dịch
FINS — chỉ cần MỘT lần ghi-0 thất bại (mạng chập chờn, board Ethernet PLC
bận) là D300 kẹt ở 1; mọi verdict PASS sau đó (cũng =1) bị coi là "không
đổi" và **bỏ qua vĩnh viễn**: không SFC, không lưu ảnh/Excel, line đứng —
nhìn như app treo dù GUI vẫn vẽ. Các fix:

1. `plc_worker.py` — handshake **ack-trước-emit-sau**: thấy ≠0 → ghi 0,
   ghi OK mới emit; ghi lỗi → poll sau đọc lại giá trị còn nguyên, thử
   lại. Không còn phụ thuộc "giá trị thay đổi". Verdict + scan-check
   (D500) đọc GỘP 1 giao dịch FINS → giảm nửa tải lên PLC.
2. `main.py` — gọi `crashlog.install()` (trước đây quên gọi → watchdog
   treo GUI + faulthandler không chạy). Giờ nếu còn treo, `crash.log`
   sẽ dump stack toàn bộ thread để truy đúng chỗ.
3. `cp2e.py` — `TCP_NODELAY` + `SO_KEEPALIVE`; **backoff 2s** sau lỗi
   connect: PLC chết không còn làm mỗi call ôm lock chờ timeout 2-6s
   (làm quét SN nghẽn hàng chục giây).
4. `scanner.py` — chế độ tắt "Quét SN" ghi D250 mỗi 0.5s thay vì 33
   lần/giây (nghẽn board Ethernet PLC); chặn buffer serial phình vô hạn
   khi scanner không gửi CR/LF.
5. `main_window.py` — bỏ decode QImage (vài chục MB/sản phẩm, chỉ để log
   tên rồi vứt); file đo dạng text chỉ đọc 256KB cuối (file phình cả
   ngày, trước đây đọc nguyên file cho MỖI verdict); cảnh báo khi job
   nền dồn ứ / upload ảnh kẹt share mạng.
