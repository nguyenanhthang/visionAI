import customtkinter as ctk
from datetime import datetime

import employees


class LoginWindow(ctk.CTk):
    def __init__(self, on_success):
        super().__init__()
        self.on_success = on_success

        self.title("Đăng nhập")
        self.geometry("420x440")
        self.resizable(False, False)

        self._center_window(420, 440)
        self._build_ui()

        self.bind("<Return>", lambda _e: self._handle_login())

    def _center_window(self, width, height):
        self.update_idletasks()
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self):
        container = ctk.CTkFrame(self, corner_radius=12)
        container.pack(expand=True, fill="both", padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(5, weight=1)

        title = ctk.CTkLabel(
            container,
            text="VISION AI",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title.grid(row=0, column=0, pady=(25, 4), padx=20, sticky="ew")

        subtitle = ctk.CTkLabel(
            container,
            text="Đăng nhập bằng mã nhân viên",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        subtitle.grid(row=1, column=0, pady=(0, 18), padx=20, sticky="ew")

        self.entry_employee_id = ctk.CTkEntry(
            container,
            placeholder_text="Mã nhân viên",
            height=38,
            font=ctk.CTkFont(size=14),
            justify="center",
        )
        self.entry_employee_id.grid(row=2, column=0, padx=40, pady=(0, 12), sticky="ew")
        self.entry_employee_id.focus()
        # Xoá highlight đỏ khi user bắt đầu gõ lại
        self.entry_employee_id.bind("<KeyPress>", lambda _e: self._clear_field_error())

        self.login_btn = ctk.CTkButton(
            container,
            text="Đăng nhập",
            height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._handle_login,
        )
        self.login_btn.grid(row=3, column=0, padx=40, pady=(0, 16), sticky="ew")

        log_header = ctk.CTkLabel(
            container,
            text="Log lỗi",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray",
            anchor="w",
        )
        log_header.grid(row=4, column=0, padx=22, pady=(0, 4), sticky="ew")

        self.error_log = ctk.CTkTextbox(
            container,
            height=110,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            fg_color=("gray90", "gray17"),
            text_color=("#b00020", "#ff6b6b"),
        )
        self.error_log.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.error_log.configure(state="disabled")

    def _log_error(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.error_log.configure(state="normal")
        self.error_log.insert("end", f"[{timestamp}] {message}\n")
        self.error_log.see("end")
        self.error_log.configure(state="disabled")

    def _flag_field_error(self):
        self.entry_employee_id.configure(border_color="#c0392b")

    def _clear_field_error(self):
        self.entry_employee_id.configure(border_color=("gray60", "gray40"))

    def _handle_login(self):
        employee_id = self.entry_employee_id.get().strip()

        if not employee_id:
            self._flag_field_error()
            self._log_error("Mã nhân viên đang để trống.")
            return

        if not employees.exists(employee_id):
            self._flag_field_error()
            self._log_error(f"Mã nhân viên '{employee_id}' không tồn tại.")
            return

        self.on_success(employee_id)
        self.destroy()
