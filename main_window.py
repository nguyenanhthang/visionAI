import queue
import customtkinter as ctk
from datetime import datetime

from plc_worker import PLCEvent, SimulatedPLCWorker


EMPLOYEE_DIRECTORY = {
    "001": "Nguyễn Văn A",
    "002": "Trần Thị B",
    "003": "Lê Văn C",
}


class MainWindow(ctk.CTk):
    PLC_DRAIN_MS = 50  # tần suất GUI lấy event từ queue (20 Hz)

    def __init__(self, employee_id):
        super().__init__()
        self.employee_id = employee_id
        self.employee_name = EMPLOYEE_DIRECTORY.get(employee_id, "Chưa xác định")
        self.product_id = "—"
        self.total_count = 0
        self.ng_count = 0
        self.ok_count = 0

        self.title("Vision AI - Giao diện chính")
        self.geometry("1400x780")
        self.minsize(1100, 640)

        self._build_layout()
        self.add_log(f"Đăng nhập thành công - Mã NV: {self.employee_id}")

        # PLC chạy trên thread riêng, đẩy event qua queue để tránh xung đột với Tk
        self.plc_queue: "queue.Queue[PLCEvent]" = queue.Queue()
        self.plc_worker = SimulatedPLCWorker(self.plc_queue, poll_interval=1.5)
        # Khi có PLC thật: thay bằng subclass PLCWorker của bạn (Modbus/S7/MELSEC...)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.plc_worker.start()
        self.after(self.PLC_DRAIN_MS, self._drain_plc_queue)

    def _build_layout(self):
        # Cột ảnh chiếm phần lớn không gian, cột thông tin/log có minsize để không co quá nhỏ
        self.grid_columnconfigure(0, weight=5, minsize=640)
        self.grid_columnconfigure(1, weight=2, minsize=340)
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

    def update_image(self, pil_image):
        """Hiển thị PIL.Image, scale vừa khít panel, giữ tỉ lệ."""
        if pil_image is None:
            self.image_label.configure(image="", text="Chưa có hình ảnh")
            return

        w = max(self.image_canvas.winfo_width() - 20, 100)
        h = max(self.image_canvas.winfo_height() - 20, 100)
        img_w, img_h = pil_image.size
        scale = min(w / img_w, h / img_h)
        size = (max(int(img_w * scale), 1), max(int(img_h * scale), 1))

        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=size)
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image  # giữ reference để tránh GC

    # ---- PLC bridge: chạy trên main thread, gọi từ after() ----

    def _drain_plc_queue(self):
        """Lấy hết event đang chờ trong queue và xử lý trên GUI thread.

        Coalesce: nếu có nhiều event 'image' liên tiếp thì chỉ giữ frame mới
        nhất - tránh GUI lag khi PLC bắn liên tục.
        """
        events = []
        try:
            while True:
                events.append(self.plc_queue.get_nowait())
        except queue.Empty:
            pass

        latest_image = None
        for event in events:
            if event.type == "image":
                latest_image = event
            else:
                self._handle_plc_event(event)
        if latest_image is not None:
            self._handle_plc_event(latest_image)

        # Tiếp tục poll nếu worker còn chạy hoặc queue chưa cạn
        if self.plc_worker.is_running() or not self.plc_queue.empty():
            self.after(self.PLC_DRAIN_MS, self._drain_plc_queue)

    def _handle_plc_event(self, event: PLCEvent):
        if event.type == "connected":
            self.add_log("PLC: đã kết nối")
        elif event.type == "disconnected":
            self.add_log("PLC: ngắt kết nối")
        elif event.type == "error":
            self.add_log(f"PLC error: {event.data}")
        elif event.type == "fatal":
            self.add_log(f"PLC fatal: {event.data}")
        elif event.type == "result":
            data = event.data or {}
            if "product_id" in data:
                self.update_product(data["product_id"])
            ok_flag = data.get("ok")
            if ok_flag is True:
                self.update_counts(ok=self.ok_count + 1, total=self.total_count + 1)
                self.add_log(f"OK - {data.get('product_id', '')}")
            elif ok_flag is False:
                self.update_counts(ng=self.ng_count + 1, total=self.total_count + 1)
                self.add_log(f"NG - {data.get('product_id', '')}")
        elif event.type == "image":
            self.update_image(event.data)

    def _on_close(self):
        self.add_log("Đang dừng PLC worker...")
        self.update_idletasks()
        self.plc_worker.stop(timeout=2.0)
        self.destroy()
