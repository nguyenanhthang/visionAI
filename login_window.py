import queue
import customtkinter as ctk
from datetime import datetime

import config
import employees
from scanner import BadgeScanner, ScanEvent


class LoginWindow(ctk.CTkToplevel):
    SCAN_DRAIN_MS = 80

    def __init__(self, master, on_success, on_close):
        super().__init__(master)
        self.on_success = on_success
        self.on_close = on_close
        self._destroyed = False
        self._drain_after_id = None

        self.title("Đăng nhập")
        self.geometry("440x460")
        self.resizable(False, False)

        self._center_window(440, 460)
        self._build_ui()

        self.bind("<Return>", lambda _e: self._handle_login())

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

        self.protocol("WM_DELETE_WINDOW", self._handle_user_close)
        self.scanner.start()
        self._schedule_drain()

        # Toplevel cần lift + focus để hiện lên trên root ẩn
        self.after(50, self._bring_to_front)

    def _bring_to_front(self):
        if self._safe():
            self.lift()
            self.focus_force()
            self.entry_employee_id.focus()

    def _safe(self) -> bool:
        """True nếu widget còn sống - guard cho mọi callback từ after()."""
        if self._destroyed:
            return False
        try:
            return bool(self.winfo_exists())
        except Exception:
            return False

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
        self.error_log.tag_config("error", foreground="#ff6b6b")
        self.error_log.tag_config("ok", foreground="#4cd964")
        self.error_log.tag_config("info", foreground="#5dade2")
        self.error_log.configure(state="disabled")

    def _log(self, message: str, tag: str = "info"):
        if not self._safe():
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.error_log.configure(state="normal")
        self.error_log.insert("end", f"[{timestamp}] {message}\n", tag)
        self.error_log.see("end")
        self.error_log.configure(state="disabled")

    def _flag_field_error(self):
        if self._safe():
            self.entry_employee_id.configure(border_color="#c0392b")

    def _clear_field_error(self):
        if self._safe():
            self.entry_employee_id.configure(border_color=("gray60", "gray40"))

    # ---- scanner bridge ----

    def _schedule_drain(self):
        if self._safe():
            self._drain_after_id = self.after(self.SCAN_DRAIN_MS, self._drain_scan_queue)

    def _drain_scan_queue(self):
        self._drain_after_id = None
        if not self._safe():
            return
        try:
            while True:
                self._handle_scan_event(self.scan_queue.get_nowait())
        except queue.Empty:
            pass
        self._schedule_drain()

    def _handle_scan_event(self, event: ScanEvent):
        if not self._safe():
            return
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

    # ---- entry points ----

    def _handle_login(self):
        if not self._safe():
            return
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
        callback = self.on_success
        self._teardown()
        if callback:
            callback(employee_id)

    def _handle_user_close(self):
        """User bấm X - cleanup rồi báo App thoát hẳn."""
        callback = self.on_close
        self._teardown()
        if callback:
            callback()

    def _teardown(self):
        """Cancel after, stop scanner, destroy window. An toàn gọi nhiều lần."""
        if self._destroyed:
            return
        self._destroyed = True

        if self._drain_after_id is not None:
            try:
                self.after_cancel(self._drain_after_id)
            except Exception:
                pass
            self._drain_after_id = None

        try:
            self.scanner.stop(timeout=1.5)
        except Exception:
            pass

        try:
            self.destroy()
        except Exception:
            pass
