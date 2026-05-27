import customtkinter as ctk
from login_window import LoginWindow
from main_window import MainWindow


class App:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.employee_id = None

    def run(self):
        login = LoginWindow(on_success=self.on_login_success)
        login.mainloop()

        if self.employee_id:
            main = MainWindow(employee_id=self.employee_id)
            main.mainloop()

    def on_login_success(self, employee_id):
        self.employee_id = employee_id


if __name__ == "__main__":
    App().run()
