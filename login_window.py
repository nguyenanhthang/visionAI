import queue
import customtkinter as ctk
from datetime import datetime

import config
import employees
from scanner import BadgeScanner, ScanEvent


class LoginWindow(ctk.CTk):
    SCAN_DRAIN_MS = 80  # tần suất GUI lấy event scanner (~12Hz)

    def __init__(self, on_success):
        super().__init__()
        self.on_success = on_success
        self._destroyed = False

        self.title("Đăng nhập")
        self.geometry("440x460")
        self.resizable(False, False)

        self._center_window(440, 460)
        self._build_ui()

        self.bind("<Return>", lambda _e: self._handle_login())

        # Scanner chạy thread riêng, đẩy event vào queue
        self.scan_queue: "queue.Queue[ScanEvent]" = queue.Queue()
        self.scanner = BadgeScanner(
            self.scan_queue,
            port=config.SCANNER_PORT,
            baudrate=config.SCANNER_BAUDRATE,
            read_size=config.SCANNER_READ_SIZE,
            serial_timeout=config.SCANNER_TIMEOUT,
            token_url=config.API_TOKEN_URL,
            employee_url_prefix=config.API_EMPLOYEE_URL_PREFIX,
            request_timeout=config.API_REQUEST_TIMEOUT,
        )

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.scanner.start()
        self.after(self.SCAN_DRAIN_MS, self._drain_scan_queue)

    def _center_window(self, width, height):
        self.update_idletasks()
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self):
        container = ctk.CTkFrame(self, corner_radius=12)
        container.pack(expand=True, fill="both", padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(6, weight=1)

        title = ctk.CTkLabel(
            container,
            text="VISION AI",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title.grid(row=0, column=0, pady=(22, 2), padx=20, sticky="ew")

        subtitle = ctk.CTkLabel(
            container,
            text="Quét thẻ hoặc nhập mã nhân viên",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        subtitle.grid(row=1, column=0, pady=(0, 14), padx=20, sticky="ew")

        self.entry_employee_id = ctk.CTkEntry(
            container,
            placeholder_text="Mã nhân viên",
            height=38,
            font=ctk.CTkFont(size=14),
            justify="center",
        )
        self.entry_employee_id.grid(row=2, column=0, padx=40, pady=(0, 10), sticky="ew")
        self.entry_employee_id.focus()
        self.entry_employee_id.bind("<KeyPress>", lambda _e: self._clear_field_error())

        self.login_btn = ctk.CTkButton(
            container,
            text="Đăng nhập",
            height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._handle_login,
        )
        self.login_btn.grid(row=3, column=0, padx=40, pady=(0, 12), sticky="ew")

        self.scanner_status = ctk.CTkLabel(
            container,
            text="● Scanner: đang khởi động...",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
        )
        self.scanner_status.grid(row=4, column=0, padx=22, pady=(0, 6), sticky="ew")

        log_header = ctk.CTkLabel(
            container,
            text="Log hệ thống",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray",
            anchor="w",
        )
        log_header.grid(row=5, column=0, padx=22, pady=(0, 4), sticky="ew")

        self.error_log = ctk.CTkTextbox(
            container,
            height=110,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            fg_color=("gray90", "gray17"),
            text_color=("gray20", "gray85"),
        )
        self.error_log.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="nsew")
        # Tag màu cho từng loại log
        self.error_log.tag_config("error", foreground="#ff6b6b")
        self.error_log.tag_config("ok", foreground="#4cd964")
        self.error_log.tag_config("info", foreground="#5dade2")
        self.error_log.configure(state="disabled")

    def _log(self, message: str, tag: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.error_log.configure(state="normal")
        self.error_log.insert("end", f"[{timestamp}] {message}\n", tag)
        self.error_log.see("end")
        self.error_log.configure(state="disabled")

    def _flag_field_error(self):
        self.entry_employee_id.configure(border_color="#c0392b")

    def _clear_field_error(self):
        self.entry_employee_id.configure(border_color=("gray60", "gray40"))

    # ---- scanner bridge: chạy trên main thread qua after() ----

    def _drain_scan_queue(self):
        try:
            while True:
                self._handle_scan_event(self.scan_queue.get_nowait())
        except queue.Empty:
            pass
        if not self._destroyed:
            self.after(self.SCAN_DRAIN_MS, self._drain_scan_queue)

    def _handle_scan_event(self, event: ScanEvent):
        if event.type == "connected":
            self.scanner_status.configure(
                text=f"● Scanner: kết nối {event.data}",
                text_color="#4cd964",
            )
            self._log(f"Scanner sẵn sàng ({event.data})", "ok")
        elif event.type == "disconnected":
            self.scanner_status.configure(
                text="● Scanner: ngắt kết nối",
                text_color="gray",
            )
        elif event.type == "scan":
            self._log(f"Đã quét: {event.data}", "info")
        elif event.type == "error":
            self._log(str(event.data), "error")
        elif event.type == "login_ok":
            data = event.data or {}
            staff_id = data.get("staff_id")
            staff_name = data.get("staff_name", "")
            if staff_id:
                self._log(f"Xác thực OK: {staff_id} - {staff_name}", "ok")
                self._finish_login(staff_id)

    # ---- manual entry ----

    def _handle_login(self):
        employee_id = self.entry_employee_id.get().strip()

        if not employee_id:
            self._flag_field_error()
            self._log("Mã nhân viên đang để trống.", "error")
            return

        if not employees.exists(employee_id):
            self._flag_field_error()
            self._log(f"Mã nhân viên '{employee_id}' không tồn tại.", "error")
            return

        self._finish_login(employee_id)

    def _finish_login(self, employee_id: str):
        self.on_success(employee_id)
        self._cleanup_and_destroy()

    def _on_close(self):
        self._cleanup_and_destroy()

    def _cleanup_and_destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        try:
            self.scanner.stop(timeout=1.5)
        finally:
            self.destroy()
