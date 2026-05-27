import customtkinter as ctk
from datetime import datetime


EMPLOYEE_DIRECTORY = {
    "001": "Nguyễn Văn A",
    "002": "Trần Thị B",
    "003": "Lê Văn C",
}


class MainWindow(ctk.CTk):
    def __init__(self, employee_id):
        super().__init__()
        self.employee_id = employee_id
        self.employee_name = EMPLOYEE_DIRECTORY.get(employee_id, "Chưa xác định")
        self.product_id = "—"
        self.total_count = 0
        self.ng_count = 0
        self.ok_count = 0

        self.title("Vision AI - Giao diện chính")
        self.geometry("1200x700")
        self.minsize(1000, 600)

        self._build_layout()
        self.add_log(f"Đăng nhập thành công - Mã NV: {self.employee_id}")

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=2)

        self._build_image_panel()
        self._build_info_panel()
        self._build_log_panel()

    def _build_image_panel(self):
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(10, 5), pady=10)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            frame,
            text="HÌNH ẢNH",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        self.image_canvas = ctk.CTkFrame(
            frame,
            fg_color=("gray85", "gray20"),
            corner_radius=8,
        )
        self.image_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.image_canvas.grid_rowconfigure(0, weight=1)
        self.image_canvas.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(
            self.image_canvas,
            text="Chưa có hình ảnh",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

    def _build_info_panel(self):
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))
        frame.grid_columnconfigure(1, weight=1)

        header = ctk.CTkLabel(
            frame,
            text="THÔNG TIN",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=15, pady=(10, 10))

        self._add_info_row(frame, 1, "Mã nhân viên:", self.employee_id)
        self._add_info_row(frame, 2, "Tên nhân viên:", self.employee_name)
        self._add_info_row(frame, 3, "Product ID:", self.product_id, store_as="product_id_label")

        separator = ctk.CTkFrame(frame, height=2, fg_color=("gray70", "gray30"))
        separator.grid(row=4, column=0, columnspan=2, sticky="ew", padx=15, pady=12)

        stats_title = ctk.CTkLabel(
            frame,
            text="SẢN LƯỢNG",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        stats_title.grid(row=5, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 8))

        stats_container = ctk.CTkFrame(frame, fg_color="transparent")
        stats_container.grid(row=6, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 15))
        stats_container.grid_columnconfigure((0, 1, 2), weight=1)

        self.total_card = self._make_stat_card(stats_container, "TOTAL", 0, "#1f6aa5")
        self.total_card.grid(row=0, column=0, sticky="nsew", padx=4)

        self.ok_card = self._make_stat_card(stats_container, "OK", 0, "#2e8b57")
        self.ok_card.grid(row=0, column=1, sticky="nsew", padx=4)

        self.ng_card = self._make_stat_card(stats_container, "NG", 0, "#c0392b")
        self.ng_card.grid(row=0, column=2, sticky="nsew", padx=4)

    def _add_info_row(self, parent, row, label_text, value_text, store_as=None):
        label = ctk.CTkLabel(
            parent,
            text=label_text,
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        label.grid(row=row, column=0, sticky="w", padx=(20, 8), pady=4)

        value = ctk.CTkLabel(
            parent,
            text=value_text,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        value.grid(row=row, column=1, sticky="ew", padx=(0, 20), pady=4)

        if store_as:
            setattr(self, store_as, value)

    def _make_stat_card(self, parent, title, value, color):
        card = ctk.CTkFrame(parent, fg_color=color, corner_radius=8, height=80)
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white",
        )
        title_label.grid(row=0, column=0, pady=(10, 0))

        value_label = ctk.CTkLabel(
            card,
            text=str(value),
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white",
        )
        value_label.grid(row=1, column=0, pady=(0, 10))

        card.value_label = value_label
        return card

    def _build_log_panel(self):
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(5, 10))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            frame,
            text="LOG HOẠT ĐỘNG",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        self.log_textbox = ctk.CTkTextbox(
            frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_textbox.configure(state="disabled")

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", line)
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def update_product(self, product_id):
        self.product_id = product_id
        self.product_id_label.configure(text=product_id)

    def update_counts(self, total=None, ok=None, ng=None):
        if total is not None:
            self.total_count = total
            self.total_card.value_label.configure(text=str(total))
        if ok is not None:
            self.ok_count = ok
            self.ok_card.value_label.configure(text=str(ok))
        if ng is not None:
            self.ng_count = ng
            self.ng_card.value_label.configure(text=str(ng))
