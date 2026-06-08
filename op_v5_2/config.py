"""Cấu hình app - chỉnh ở đây thay vì hardcode trong code.

Giá trị trong file này là DEFAULT. settings.json cùng folder sẽ
override khi app khởi động (xem settings_window.load_settings_overrides).
"""

# ── Barcode scanner (badge reader) ──────────────────────────
SCANNER_PORT       = 'COM13'
SCANNER_BAUDRATE   = 115200
SCANNER_READ_SIZE  = 8       # số byte tối đa cho 1 lần đọc badge
SCANNER_TIMEOUT    = 0.5     # giây - càng nhỏ stop càng nhanh
PRODUCT_READ_SIZE  = 64      # số byte tối đa cho 1 lần đọc mã sản phẩm
Staff              = ''  # mã NV mặc định

# ── Bật/tắt tính năng (toggle ở giao diện chính) ───────────
SCAN_ENABLED = True    # bật quét SN sản phẩm (bắt buộc quét trước khi qua trạm)
SAVE_EXCEL   = True    # bật lưu dòng đo sang file .xls
SAVE_IMAGE   = True    # bật lưu/đẩy ảnh OPL

# ── API xác thực nhân viên ─────────────────────────────────
# Để rỗng → fallback dùng employees.py local.
API_TOKEN_URL           = ""
API_EMPLOYEE_URL_PREFIX = ""
API_REQUEST_TIMEOUT     = 5  # giây

# ── PLC (Omron CP2E qua FINS/TCP) ─────────────────────────
PLC_IP                = '192.168.250.1'  # IP PLC
PLC_PORT              = 9600             # cổng FINS/TCP (Omron mặc định 9600)
PLC_RESULT_ADDR       = 300              # DM word AOI ghi verdict: 1=OK, 2=NG  (D300)
PLC_SCAN_RESULT_ADDR  = 250              # DM word Scanner ghi sau khi check SFC (D250)
PLC_SCAN_CHECK_ADDR   = 500              # DM PLC bật =1 để yêu cầu kiểm tra "đã quét SN chưa" (D500)
PLC_POLL_HZ           = 5                # số lần poll PLC / giây
PLC_SIMULATED         = False            # True → chạy giả lập, False → đọc PLC thật
# ── Station / SFC ──────────────────────────────────────────
STATION_NAME = 'AOI-C219B-S1'
ON_OFF_SFC   = False                     # True → push clipThroughStation; False → bỏ qua
sn_link1 = 'http://10.222.48.213:8888/v2/pass/mes/tsc/check/TSC-VN/tsc_vn1/AOI-C219B-S1?sn='
sn_link2 = '&station_id=AOI-C219B-S1'
link_post_img = '//10.222.48.222/cdpaoi/AOI-C219B-S1/'
link_sfc = 'http://10.222.48.213:8888/v2/pass/mes/tsc/tsc_vn1/clipThroughStation'
emp_link = 'http://10.222.48.213:8888/v2/platform/staff-detail?factoryCode=TSC_VN&staffCode='
token_link = 'http://10.222.48.213:8888/v2/platform/get/token?appid=e015ef3a23a842419a6a36373f9db9b8&appsecret=405a03d085a1406dbfb74ee941de2c6e&transid=10005999927000000062014101615303080000001'

# ── OPL attachments (ảnh kiểm tra) ─────────────────────────
# Ảnh được hệ thống AOI chia làm 2 cây:
#   <OPL_OK_DIR>\<YYYY-MM-DD>\<image>.png  — sản phẩm OK
#   <OPL_NG_DIR>\<YYYY-MM-DD>\<image>.png  — sản phẩm NG
# Khi PLC reg 300 = 1 → lấy ảnh từ OK, = 2 → lấy từ NG.
# Submit/auto-trigger sẽ chọn subfolder có mtime mới nhất rồi
# lấy file ảnh có mtime mới nhất trong subfolder đó.
OPL_OK_DIR = 'D:\\anh\\OK'
OPL_NG_DIR = 'D:\\anh\\NG'

# ── Data export (.xls log) ─────────────────────────────────
# Mỗi tín hiệu PLC OK/NG: đọc dòng dữ liệu MỚI NHẤT (cột B→I = 8 giá trị)
# của file  <DATA_SRC_DIR>\<ngày>.xls  rồi append 1 dòng
#   [times, SN, L1-1, L1-2, L2-1, L2-2, L3-1, L3-2, L4-1, L4-2, result]  (result = OK/NG)
# sang  <DATA_EXPORT_DIR>\<ngày>.xls  (header tạo 1 lần).
# Để rỗng DATA_SRC_DIR hoặc DATA_EXPORT_DIR → tắt tính năng này.
DATA_SRC_DIR      = 'D:/data'          # folder chứa file .xls nguồn (đặt tên theo ngày)
DATA_EXPORT_DIR   = 'D:/AOI_DATA'          # folder lưu file .xls đích (đặt tên theo ngày)
DATA_FILE_DATEFMT = '%Y-%m-%d'    # định dạng tên file theo ngày (vd 20260603.xls)
