import customtkinter as ctk

from login_window import LoginWindow
from main_window import MainWindow


class App:
    """1 root ẩn + 2 Toplevel (login, main).

    Tránh tạo nhiều ctk.CTk() liên tiếp - pattern này gây ra lỗi
    'invalid command name ...' khi Tcl interpreter cũ chưa cleanup
    xong mà cái mới đã chạy.
    """

    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.withdraw()  # ẩn root, chỉ dùng làm container

        self._login = None
        self._main = None

    def run(self):
        self._login = LoginWindow(
            self.root,
            on_success=self._open_main,
            on_close=self._exit,
        )
        self.root.mainloop()

    def _open_main(self, employee_id):
        # Tạo cửa sổ chính SAU KHI login đã destroy hoàn toàn
        # Dùng after_idle để defer ra khỏi callback hiện tại
        self.root.after_idle(lambda: self._spawn_main(employee_id))

    def _spawn_main(self, employee_id):
        self._main = MainWindow(
            self.root,
            employee_id=employee_id,
            on_close=self._exit,
        )

    def _exit(self):
        try:
            self.root.quit()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    App().run()
