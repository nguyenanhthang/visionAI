"""Danh bạ nhân viên - dùng chung cho login và main window."""

EMPLOYEE_DIRECTORY = {
    "V318048": "LUONG THI KIEN",
    "002": "Trần Thị B",
    "003": "Lê Văn C",
}


def lookup(employee_id: str):
    return EMPLOYEE_DIRECTORY.get(employee_id)


def exists(employee_id: str) -> bool:
    return employee_id in EMPLOYEE_DIRECTORY
