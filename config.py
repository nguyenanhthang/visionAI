"""Cấu hình app - chỉnh ở đây thay vì hardcode trong code."""

# Barcode scanner (badge reader)
SCANNER_PORT = "COM4"
SCANNER_BAUDRATE = 9600
SCANNER_READ_SIZE = 8       # số byte tối đa cho 1 lần đọc
SCANNER_TIMEOUT = 0.5       # giây - càng nhỏ stop càng nhanh

# API xác thực nhân viên. Để rỗng → fallback dùng employees.py local.
API_TOKEN_URL = ""
API_EMPLOYEE_URL_PREFIX = ""
API_REQUEST_TIMEOUT = 5     # giây
