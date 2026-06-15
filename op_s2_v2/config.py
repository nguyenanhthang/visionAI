"""Cấu hình app - chỉnh ở đây thay vì hardcode trong code.

Giá trị trong file này là DEFAULT. settings.json cùng folder sẽ
override khi app khởi động (xem settings_window.load_settings_overrides).
"""

# ── Barcode scanner (badge reader) ──────────────────────────
SCANNER_PORT       = 'COM2'
SCANNER_BAUDRATE   = 9600
SCANNER_READ_SIZE  = 8       # số byte tối đa cho 1 lần đọc badge
SCANNER_TIMEOUT    = 0.5     # giây - càng nhỏ stop càng nhanh
PRODUCT_READ_SIZE  = 64      # số byte tối đa cho 1 lần đọc mã sản phẩm
Staff              = ''  # mã NV mặc định

# ── API xác thực nhân viên ─────────────────────────────────
# Để rỗng → fallback dùng employees.py local.
API_TOKEN_URL           = ""
API_EMPLOYEE_URL_PREFIX = ""
API_REQUEST_TIMEOUT     = 5  # giây
SCAN_ENABLED = True    # bật quét SN sản phẩm (bắt buộc quét trước khi qua trạm)
SAVE_EXCEL   = True    # bật lưu dòng đo sang file .xls
SAVE_IMAGE   = True    # bật lưu/đẩy ảnh OPL
# Chặn tool accessibility (UIA/MSAA — remote desktop/AV/agent giám sát…)
# attach vào app. Client treo từng làm GUI đơ cứng ngay trong lệnh ghi log
# (crash.log op_v7 Timeout 11/06). App kiosk không cần screen-reader → chặn.
DISABLE_ACCESSIBILITY = True
# ── PLC (Omron CP2E qua FINS/TCP) ─────────────────────────
PLC_IP                = '192.168.250.1'  # IP PLC
PLC_PORT              = 9600             # cổng FINS/TCP (Omron mặc định 9600)
PLC_RESULT_ADDR       = 300              # DM word AOI ghi verdict: 1=OK, 2=NG  (D300)
PLC_SCAN_RESULT_ADDR  = 250              # DM word Scanner ghi sau khi check SFC (D250)
PLC_POLL_HZ           = 5                # số lần poll PLC / giây
PLC_SIMULATED         = False            # True → chạy giả lập, False → đọc PLC thật
# ── Station / SFC ──────────────────────────────────────────
STATION_NAME = 'AOI-S2-LINE10'
ON_OFF_SFC   = True                     # True → push clipThroughStation; False → bỏ qua
sn_link1 = 'http://10.222.48.213:8888/v2/pass/mes/tsc/check/TSC-VN/tsc_vn1/AOI-S2-LINE10?sn='
sn_link2 = '&station_id=AOI-S2-LINE10'
link_post_img = '//10.222.48.222/cdpaoi/AOI-S2-LINE10/'
link_sfc = 'http://10.222.48.213:8888/v2/pass/mes/tsc/tsc_vn1/clipThroughStation'
emp_link = 'http://10.222.48.213:8888/v2/platform/staff-detail?factoryCode=TSC_VN&staffCode='
token_link = 'http://10.222.48.213:8888/v2/platform/get/token?appid=e015ef3a23a842419a6a36373f9db9b8&appsecret=405a03d085a1406dbfb74ee941de2c6e&transid=10005999927000000062014101615303080000001'

# ── OPL attachments (ảnh kiểm tra) ─────────────────────────
# Máy AOI lưu 2 ảnh của 1 sản phẩm vào 2 cây riêng:
#   <OPL_INSIDE_DIR>\<folder mới nhất>\<ảnh mới nhất>   — mặt trong
#   <OPL_OUTSIDE_DIR>\<folder mới nhất>\<ảnh mới nhất>  — mặt ngoài
# Mỗi tín hiệu D300: lấy ảnh mới nhất ở CẢ 2 cây (folder con + ảnh đều
# theo mtime mới nhất) → GỘP NGANG (inside trái · outside phải) thành 1
# ảnh → đẩy lên link_post_img. OK/NG do D300 (1=OK, 2=NG) chỉ quyết NHÃN
# tên file (Passed/Failed), KHÔNG chọn folder.
OPL_INSIDE_DIR  = 'E:/Images/Graphics/inside'
OPL_OUTSIDE_DIR = 'E:/Images/Graphics/outside'

# ── Data export (.xls log) ─────────────────────────────────
# Mỗi tín hiệu PLC OK/NG: đọc dòng dữ liệu MỚI NHẤT (cột B→F = 5 giá trị)
# của file  <DATA_SRC_DIR>\<ngày>.xls  (hoặc .csv) rồi append 1 dòng
#   [times, SN, Yellow, Orange, Black, Red, OK NG, result]  (result = OK/NG)
# sang  <DATA_EXPORT_DIR>\<ngày>.xls  (header tạo 1 lần).
# Để rỗng DATA_SRC_DIR hoặc DATA_EXPORT_DIR → tắt tính năng này.
DATA_SRC_DIR      = 'E:/CSV'          # folder chứa file .xls nguồn (đặt tên theo ngày)
DATA_EXPORT_DIR   = 'E:/AOI_DATA'          # folder lưu file .xls đích (đặt tên theo ngày)
DATA_FILE_DATEFMT = '%Y-%m-%d'    # định dạng tên file theo ngày (vd 20260603.xls)
