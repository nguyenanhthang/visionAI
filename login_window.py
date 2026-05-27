import customtkinter as ctk
from tkinter import messagebox


class LoginWindow(ctk.CTk):
    def __init__(self, on_success):
        super().__init__()
        self.on_success = on_success

        self.title("Đăng nhập")
        self.geometry("400x300")
        self.resizable(False, False)

        self._center_window(400, 300)
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

        title = ctk.CTkLabel(
            container,
            text="VISION AI",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title.pack(pady=(30, 5))

        subtitle = ctk.CTkLabel(
            container,
            text="Đăng nhập bằng mã nhân viên",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        subtitle.pack(pady=(0, 20))

        self.entry_employee_id = ctk.CTkEntry(
            container,
            placeholder_text="Mã nhân viên",
            width=260,
            height=38,
            font=ctk.CTkFont(size=14),
            justify="center",
        )
        self.entry_employee_id.pack(pady=(0, 15))
        self.entry_employee_id.focus()

        login_btn = ctk.CTkButton(
            container,
            text="Đăng nhập",
            width=260,
            height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._handle_login,
        )
        login_btn.pack()

    def _handle_login(self):
        employee_id = self.entry_employee_id.get().strip()
        if not employee_id:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập mã nhân viên.")
            return

        self.on_success(employee_id)
        self.destroy()
