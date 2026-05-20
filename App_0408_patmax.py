"""
App_0408_patmax.py — App AOI TESLA bản chuyển từ Faster R-CNN (torch) sang
PatQuick (CogPatMax pattern match) của VisionPro/core/patmax_engine.

Khác biệt vs App_0408.py:
  • __init__ không load `model_screw1.pth` nữa. Load PatMax model:
        model.json + model.npz  (cùng folder, đặt tên `model.json`)
  • Hàm AOI(): bỏ torch inference, dùng `run_patmax_align(algorithm="PatQuick")`
    để tìm screws. count = số results có score ≥ accept_threshold.
  • Logic OK/NG giữ nguyên: count == 3 + 3 moment trong [0.4, 0.6] → OK.
  • Bbox vẽ từ result.x/y/width/height. Score label kế bbox.

Cần: thư mục `VisionPro/` (chứa core/patmax_engine.py) ở cùng cấp với file
script HOẶC trong PYTHONPATH. Model files (`model.json` + `model.npz`)
mặc định đặt ngay cạnh script (override qua _PATMAX_MODEL_PATH).
"""
import os
import sys
from time import sleep
import customtkinter
import tkinter
from PIL import Image, ImageTk
from datetime import datetime, date
import cv2
import serial
import numpy as np
from INI import Ini
from tkinter import messagebox
import json


# ── Wire VisionPro/core/patmax_engine vào sys.path ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# Thứ tự thử: cùng folder, sibling "VisionPro", parent, parent/VisionPro
for _cand in (_HERE,
               os.path.join(_HERE, "VisionPro"),
               os.path.dirname(_HERE),
               os.path.join(os.path.dirname(_HERE), "VisionPro")):
    if os.path.isfile(os.path.join(_cand, "core", "patmax_engine.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

from core.patmax_engine import load_model, run_patmax_align, PatMaxModel

# Path tới model PatMax (cùng folder script mặc định). load_model() tự tìm
# cả .json lẫn .npz cùng base.
_PATMAX_MODEL_PATH = os.path.join(_HERE, "model.json")

# ── PatQuick search tuning ──────────────────────────────────────────
# Model.json train với angle range = 0 (pattern 1 góc cố định). NHƯNG vít
# lục giác trên sản phẩm xoay tự do — mỗi con 1 hướng → nếu search không
# quét góc thì chỉ match đúng con vít cùng góc với pattern, các con khác
# score thấp → bị loại. Bật quét góc ở search time để tìm đủ.
#   _SEARCH_ANGLE_RANGE : quét ±deg quanh 0. Hex head đối xứng 60° nên 60
#                         là đủ; để 180 cho chắc. Đặt 0 = tắt quét góc.
#   _SEARCH_ANGLE_STEP  : bước quét (deg). Nhỏ hơn = nhạy hơn nhưng chậm.
#   _SEARCH_ACCEPT_OVERRIDE : None = dùng accept_threshold của model.
#                         Đặt số 0..1 (vd 0.6) nếu vài vít score hơi thấp
#                         do lighting/washer khác — hạ ngưỡng để bắt thêm.
_SEARCH_ANGLE_RANGE = 180.0
_SEARCH_ANGLE_STEP = 12.0
_SEARCH_ACCEPT_OVERRIDE = None
# Diagnostic: vẽ luôn candidate DƯỚI ngưỡng (màu cam + score) lên ảnh để
# user thấy vít bị "trượt" score bao nhiêu → biết hạ _SEARCH_ACCEPT_OVERRIDE
# xuống mức nào. Tắt (False) khi đã tune xong cho production.
_SEARCH_DIAG = True
_SEARCH_DIAG_FLOOR = 0.35   # score tối thiểu để vẽ candidate diagnostic


def _select_ellipse_roi(window_name: str, img: np.ndarray):
    """Custom ellipse ROI selector — drag chuột vẽ bbox; ellipse nội tiếp
    được render live theo dõi cursor. Trả (x, y, w, h) khi Enter,
    (0,0,0,0) khi Esc/cancel.

    Khác cv2.selectROI ở chỗ HÌNH HIỂN THỊ là ellipse, khớp với mask thật
    PatMax dùng (shape_type='ellipse' = inscribed ellipse trong bbox).

    Controls:
      Left-drag    → vẽ ellipse
      Enter/Space  → confirm
      r            → reset (vẽ lại)
      Esc / c      → cancel
    """
    state = {"active": False, "start": None, "end": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["active"] = True
            state["start"] = (x, y); state["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["active"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["active"] = False
            state["end"] = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    HINT = "Drag chuot ve ellipse | Enter=OK  r=reset  Esc/c=cancel"
    while True:
        disp = img.copy()
        s, e = state["start"], state["end"]
        if s is not None and e is not None:
            x1, y1 = s; x2, y2 = e
            cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
            rw = abs(x2 - x1) // 2; rh = abs(y2 - y1) // 2
            if rw >= 1 and rh >= 1:
                # Bbox xám mờ + ellipse xanh — user thấy rõ vùng match thật
                cv2.rectangle(disp, (min(x1, x2), min(y1, y2)),
                              (max(x1, x2), max(y1, y2)),
                              (120, 120, 120), 1)
                cv2.ellipse(disp, (cx, cy), (rw, rh), 0, 0, 360,
                            (0, 255, 0), 2)
        cv2.putText(disp, HINT, (10, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow(window_name, disp)
        k = cv2.waitKey(15) & 0xFF
        if k in (13, 32):   # Enter / Space
            if s is not None and e is not None:
                x = min(s[0], e[0]); y = min(s[1], e[1])
                w = abs(e[0] - s[0]); h = abs(e[1] - s[1])
                if w > 1 and h > 1:
                    cv2.destroyWindow(window_name)
                    return (x, y, w, h)
        elif k == ord('r'):
            state["start"] = state["end"] = None
            state["active"] = False
        elif k in (27, ord('c')):   # Esc / c
            cv2.destroyWindow(window_name)
            return (0, 0, 0, 0)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title('AOI TESLA')
        self.state('zoomed')
        self.update()
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.sys = Ini('sys.ini')
        self.sys_quantity = Ini('sys_quantity.ini')
        self.count_OK = int(self.sys_quantity.find('OK'))
        self.count_NG = int(self.sys_quantity.find('NG'))
        self.count_Total = self.count_OK + self.count_NG
        self.re_screws = int(self.sys.find('remainning_screws'))
        self.quantity_screws = int(self.sys.find('quantity_screws'))
        self.staff_id = self.sys.find('staff_id')
        self.job = self.sys.find('job')
        self.on_SFC = self.sys.find('on_off_sfc')
        self.scan = self.sys.find('SAN_MOTOR')
        self.COM_LEFT = self.sys.find('SCAN_LEFT')
        self.COM_RIGHT = self.sys.find('SCAN_RIGHT')
        self.COM_MOTOR = self.sys.find('SAN_MOTOR')

        self.frame_top = customtkinter.CTkFrame(master=self, width=200, corner_radius=10, fg_color='sky blue')
        self.frame_top.grid(row=0, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)

        self.frame_left = customtkinter.CTkFrame(master=self, width=200, corner_radius=10, fg_color='sky blue')
        self.frame_left.grid(row=1, column=0, sticky="nswe", padx=5, pady=5)

        self.frame_right = customtkinter.CTkFrame(master=self, corner_radius=10, fg_color='sky blue')
        self.frame_right.grid(row=1, column=1, sticky="nswe", padx=5, pady=5)

        self.canvas_L = customtkinter.CTkCanvas(self.frame_left, bg="sky blue")
        self.canvas_L.pack(expand=True, fill=customtkinter.BOTH, padx=10, pady=10)

        self.lbl_canvas_L = customtkinter.CTkLabel(self.canvas_L, text='Waitting...', font=customtkinter.CTkFont('Times new roman', 32, 'bold'))

        self.canvas_R = customtkinter.CTkCanvas(self.frame_right, bg="sky blue")
        self.canvas_R.pack(expand=True, fill=customtkinter.BOTH, padx=10, pady=10)

        self.lbl_canvas_R = customtkinter.CTkLabel(self.canvas_R, text='Waitting...', font=customtkinter.CTkFont('Times new roman', 32, 'bold'))

        self.lbl_name_app = customtkinter.CTkLabel(self.frame_top, text="AOI TESlA(3)", anchor='center', font=customtkinter.CTkFont('Times new roman', 32, 'bold'))
        self.lbl_name_app.grid(row=0, column=0, columnspan=6, padx=(100, 0), pady=10)

        self.lbl_name_staffid = customtkinter.CTkLabel(self.frame_top, text='Employee ID: ' + self.staff_id, font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_name_staffid.grid(row=1, column=0, padx=(100, 10), pady=0)

        self.btn_name_staffid = customtkinter.CTkButton(self.frame_top, text='Date: ' + self.job, font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.btn_name_staffid.grid(row=2, column=0, padx=(100, 10), pady=0)

        self.lbl_product_id = customtkinter.CTkLabel(self.frame_top, text='Product ID', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_product_id.grid(row=1, column=1, padx=(0, 150), pady=0)

        self.btn_product_id = customtkinter.CTkButton(self.frame_top, width=250, text='', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.btn_product_id.grid(row=2, column=1, padx=10, pady=0)

        self.lbl_motor_id = customtkinter.CTkLabel(self.frame_top, text='Motor ID', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_motor_id.grid(row=1, column=2, padx=(0, 150), pady=0)

        self.btn_motor_id = customtkinter.CTkButton(self.frame_top, width=250, text='', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.btn_motor_id.grid(row=2, column=2, padx=10, pady=0)

        self.lbl_nof_sfc = customtkinter.CTkLabel(self.frame_top, text='Notification SFC', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_nof_sfc.grid(row=1, column=3, padx=(0, 80), pady=0)

        self.btn_nof_sfc = customtkinter.CTkButton(self.frame_top, width=250, text='', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.btn_nof_sfc.grid(row=2, column=3, padx=10, pady=0)

        self.lbl_nof = customtkinter.CTkLabel(self.frame_top, text='Notification', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_nof.grid(row=1, column=4, padx=(0, 80), pady=0)

        self.btn_nof = customtkinter.CTkButton(self.frame_top, width=250, text='', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.btn_nof.grid(row=2, column=4, padx=10, pady=0)

        self.lbl_optionmenu = customtkinter.CTkLabel(self.frame_top, text='Select Option', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_optionmenu.grid(row=1, column=5, padx=(0, 30), pady=0)

        self.option_menu = customtkinter.CTkOptionMenu(self.frame_top, values=['Reset Sản Lượng', 'Thiết lập ROIs Right', 'Thiết lập ROIs Left', 'Train PatMax', 'Motor cu', 'Motor moi'], command=self.st_option_menu)
        self.option_menu.grid(row=2, column=5, padx=10, pady=10)

        self.lbl_quantity_screws = customtkinter.CTkLabel(self.frame_top, text='Quantity Screws', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_quantity_screws.grid(row=3, column=0, columnspan=6, padx=(800, 10), pady=(0, 80))

        self.btn_quantity_screws = customtkinter.CTkButton(self.frame_top, width=100, text=self.quantity_screws, font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.btn_quantity_screws.grid(row=3, column=0, columnspan=6, padx=(800, 10), pady=(0, 0))

        self.lbl_remainning_screws = customtkinter.CTkLabel(self.frame_top, text='Remainning Screws', font=customtkinter.CTkFont('Times new roman', 16, 'normal'))
        self.lbl_remainning_screws.grid(row=3, column=0, columnspan=6, padx=(1100, 10), pady=(0, 80))

        self.btn_remainning_screws = customtkinter.CTkButton(self.frame_top, width=100, text=self.re_screws, font=customtkinter.CTkFont('Times new roman', 16, 'normal'), command=lambda: self.display_canvas_R(0.5, 0.5, 0.5))
        self.btn_remainning_screws.grid(row=3, column=0, columnspan=6, padx=(1100, 10), pady=(0, 0))

        self.lbl_OK_NG = customtkinter.CTkLabel(self.frame_top, width=250, height=150, text='...', font=customtkinter.CTkFont('Times new roman', 64, 'bold'))
        self.lbl_OK_NG.grid(row=3, column=0, columnspan=6, padx=(10, 600), pady=(30, 10))

        self.lbl_L_R = customtkinter.CTkLabel(self.frame_top, width=250, height=150, text='LOADING..', font=customtkinter.CTkFont('Times new roman', 48, 'bold'), text_color='green')
        self.lbl_L_R.grid(row=3, column=0, columnspan=6, padx=(120, 10), pady=(30, 10))

        self.count_total = customtkinter.CTkLabel(self.frame_top, width=150, text='Total: ' + str(self.count_Total), fg_color='white', font=customtkinter.CTkFont('Times new roman', 16, 'normal'), corner_radius=10)
        self.count_total.grid(row=3, column=0, columnspan=6, padx=(0, 1100), pady=(0, 100))
        self.count_ok = customtkinter.CTkLabel(self.frame_top, width=150, text='OK: ' + str(self.count_OK), fg_color='white', text_color='green', font=customtkinter.CTkFont('Times new roman', 16, 'normal'), corner_radius=10)
        self.count_ok.grid(row=3, column=0, columnspan=6, padx=(0, 1100), pady=(0, 30))
        self.count_ng = customtkinter.CTkLabel(self.frame_top, width=150, text='NG: ' + str(self.count_NG), fg_color='white', text_color='red', font=customtkinter.CTkFont('Times new roman', 16, 'normal'), corner_radius=10)
        self.count_ng.grid(row=3, column=0, columnspan=6, padx=(0, 1100), pady=(40, 0))

        self.on_off_SFC = customtkinter.CTkSwitch(self.frame_top, text="ON/OFF SFC", command=self.state_on_off_SFC)
        self.on_off_SFC.grid(row=3, column=0, columnspan=6, padx=(1400, 0), pady=(0, 0))

        try:
            self.ser_R = serial.Serial(self.COM_RIGHT, 115200, 8, timeout=1)
            self.ser_L = serial.Serial(self.COM_LEFT, 115200, 8, timeout=1)
            self.ser_motor = serial.Serial(self.COM_MOTOR, 9600, 8, timeout=1)
            self.sys.replace('True', 'check_scan')
        except Exception:
            self.sys.replace('False', 'check_scan')

        state_on = customtkinter.BooleanVar(self, value=True)
        state_off = customtkinter.BooleanVar(self, value=False)
        if self.on_SFC is True:
            self.on_off_SFC.configure(variable=state_on)
        else:
            self.on_off_SFC.configure(variable=state_off)
        self.update()

        self.video = cv2.VideoCapture(0)
        self.video.set(3, 4608)
        self.video.set(4, 3456)

        # ── PatMax model loading (thay cho Faster R-CNN .pth) ──────
        # load_model() đọc cả `<base>.json` (meta) và `<base>.npz` (patches +
        # edge image). Trả None nếu file vắng → app vẫn chạy, AOI() báo NG
        # toàn bộ + ghi notification.
        self.patmax_model: PatMaxModel = load_model(_PATMAX_MODEL_PATH)
        if self.patmax_model is None or not self.patmax_model.is_valid():
            self.patmax_model = None
            try:
                messagebox.showerror(
                    "PatMax Model",
                    f"Không load được model PatMax tại:\n{_PATMAX_MODEL_PATH}\n\n"
                    "Cần 2 file: model.json + model.npz cùng folder.")
            except Exception:
                pass
            self.btn_nof.configure(text='PatMax model missing',
                                     text_color='red')

    def st_option_menu(self, value):
        if value == 'Reset Sản Lượng':
            self.count_OK = 0
            self.count_NG = 0
            self.count_Total = 0
            self.sys.replace('0', 'OK')
            self.sys.replace('0', 'NG')
            self.count_ok.configure(text='OK: 0')
            self.count_ng.configure(text='NG: 0')
            self.count_total.configure(text='Total: 0')
        elif value == 'Thiết lập ROIs Right':
            sleep(0.5)
            while True:
                ret, frame = self.video.read()
                frame = cv2.resize(frame, (880, 680))
                cv2.imwrite(r"img/roi_r.png", frame)
                break
            img = cv2.imread("img/roi_r.png")
            ROIs = cv2.selectROIs("Select Rois", img)
            r = messagebox.askquestion('Notification', 'Bạn có muốn lưu lại không?')
            if r == 'yes':
                for idx, name in enumerate(("roi1_R.ini", "roi2_R.ini", "roi3_R.ini")):
                    with open(name, "w") as file:
                        file.write(str(ROIs[idx][0]) + "\n")
                        file.write(str(ROIs[idx][1]) + "\n")
                        file.write(str(ROIs[idx][2]) + "\n")
                        file.write(str(ROIs[idx][3]))
        elif value == 'Thiết lập ROIs Left':
            while True:
                ret, frame = self.video.read()
                frame = cv2.resize(frame, (880, 680))
                cv2.imwrite(r"img/roi_l.png", frame)
                break
            img = cv2.imread("img/roi_l.png")
            ROIs = cv2.selectROIs("Select Rois", img)
            r = messagebox.askquestion('Notification', 'Bạn có muốn lưu lại không?')
            if r == 'yes':
                for idx, name in enumerate(("roi1_L.ini", "roi2_L.ini", "roi3_L.ini")):
                    with open(name, "w") as file:
                        file.write(str(ROIs[idx][0]) + "\n")
                        file.write(str(ROIs[idx][1]) + "\n")
                        file.write(str(ROIs[idx][2]) + "\n")
                        file.write(str(ROIs[idx][3]))
        elif value == 'Train PatMax':
            self._train_patmax_interactive()

    def _train_patmax_interactive(self):
        """Train PatMax từ frame camera hiện tại:
          1. Capture frame → save img/train.png
          2. cv2.selectROI để user vẽ box quanh 1 con vít
          3. Confirm → train_patmax + save_model (override model.json+.npz)
          4. Reload self.patmax_model — AOI() lần sau dùng model mới

        Shape mặc định 'ellipse' khớp với model gốc (vít tròn). Search params
        (accept_threshold, num_results, overlap_threshold) GIỮ từ model cũ
        nếu có — train mới không reset tuning của user."""
        from core.patmax_engine import train_patmax, save_model
        # 1. Capture frame
        sleep(0.5)
        frame = None
        for _ in range(4):
            ret, frame = self.video.read()
        if frame is None:
            messagebox.showerror('Train PatMax', 'Không lấy được frame camera.')
            return
        frame = cv2.resize(frame, (880, 680))
        os.makedirs("img", exist_ok=True)
        cv2.imwrite("img/train.png", frame)
        img = cv2.imread("img/train.png")

        # 2. User pick ROI (ellipse direct — preview live khớp với mask thật
        # PatMax dùng; bbox vẫn được lưu dưới dạng rect cho tương thích).
        roi = _select_ellipse_roi(
            "Train PatMax — drag ve ellipse quanh con vit", img)
        x, y, w, h = (int(v) for v in roi)
        if w <= 0 or h <= 0:
            return  # user cancelled

        # 3. Confirm
        ans = messagebox.askquestion(
            'Train PatMax',
            f'Train ellipse ROI ({x}, {y}) kích thước {w}×{h}?\n'
            f'Mask: ellipse nội tiếp bbox.\n'
            f'Model sẽ ghi đè {_PATMAX_MODEL_PATH}.')
        if ans != 'yes':
            return

        # 4. Train (mode "evaluate" như model gốc) — preserve search tuning
        old = self.patmax_model
        accept_thr = old.accept_threshold if old else 0.7
        num_res    = old.num_results      if old else 3
        overlap    = old.overlap_threshold if old else 0.5
        ang_lo     = old.angle_low        if old else 0.0
        ang_hi     = old.angle_high       if old else 0.0
        ang_st     = old.angle_step       if old else 5.0
        sc_lo      = old.scale_low        if old else 1.0
        sc_hi      = old.scale_high       if old else 1.0
        canny_lo   = old.canny_low        if old else 50
        canny_hi   = old.canny_high       if old else 150

        try:
            new_model = train_patmax(
                img, (x, y, w, h),
                shape_type='ellipse',
                shape_data={'x': x, 'y': y, 'w': w, 'h': h},
                train_mode='evaluate',
                canny_low=canny_lo, canny_high=canny_hi,
                angle_low=ang_lo, angle_high=ang_hi, angle_step=ang_st,
                scale_low=sc_lo, scale_high=sc_hi,
            )
        except Exception as ex:
            messagebox.showerror('Train PatMax', f'Train failed:\n{ex}')
            return

        # Restore search tuning từ model cũ
        new_model.accept_threshold = accept_thr
        new_model.num_results      = num_res
        new_model.overlap_threshold = overlap

        # 5. Save + reload
        try:
            save_model(new_model, _PATMAX_MODEL_PATH)
        except Exception as ex:
            messagebox.showerror('Train PatMax', f'Save failed:\n{ex}')
            return

        self.patmax_model = new_model
        msg = (f'Train + save OK.\n'
                f'  Edges: {new_model.edge_count}\n'
                f'  Hash:  {new_model.model_hash}\n'
                f'  Accept: {accept_thr}  ·  Num results: {num_res}\n'
                f'  File:  {_PATMAX_MODEL_PATH}')
        messagebox.showinfo('Train PatMax', msg)
        self.btn_nof.configure(
            text=f'PatMax trained (edges={new_model.edge_count})',
            text_color='green')

    def state_on_off_SFC(self):
        if self.on_SFC is True:
            self.on_SFC = False
            self.sys.replace('False', 'on_off_sfc')
        else:
            self.on_SFC = True
            self.sys.replace('True', 'on_off_sfc')

    def Lock_QC(self):
        lock = self.sys.find('Lock_QC')
        check_scan = self.sys.find('check_scan')
        self.lbl_L_R.configure(text='LOCK QC', text_color='red')
        try:
            if check_scan is True:
                while True:
                    if self.ser_motor.isOpen():
                        self.ser_motor.write(bytearray(b'\x01\x54\x04'))
                    else:
                        self.ser_motor.open()
                        self.ser_motor.write(bytearray(b'\x01\x54\x04'))
                    data = self.ser_motor.readline(100)
                    data_scan = str(data)[2:-5]
                    value_product_id = data_scan
                    if len(value_product_id) > 0:
                        if value_product_id == lock:
                            self.lbl_L_R.configure(text='LOADING..', text_color='black')
                            self.ser_motor.close()
                            return True
                        else:
                            self.btn_nof.configure(text='Sai mã QC', text_color='red')
            else:
                self.btn_nof.configure(text='Khong ket noi Scan', text_color='red')
                self.lbl_L_R.configure(text='LOADING..', text_color='black')
                return True
        except Exception as ex:
            self.btn_nof.configure(text='Error QC: {}'.format(str(ex)), text_color='red')

    def _patmax_detect_screws(self, img_bgr: np.ndarray):
        """Tìm screws bằng PatQuick. Trả (accepted, rejected) — 2 list dict
        {x,y,w,h,score,angle}.

        • Quét rotation (_SEARCH_ANGLE_RANGE) vì model train angle 0-0 nhưng
          vít hex trên sản phẩm xoay tự do — không quét góc thì chỉ bắt
          được 1 con cùng hướng pattern.
        • accepted = score ≥ ngưỡng (đếm vào count). rejected = candidate
          dưới ngưỡng nhưng ≥ _SEARCH_DIAG_FLOOR — để vẽ diagnostic giúp
          user thấy vít "trượt" bao nhiêu mà chỉnh ngưỡng."""
        if self.patmax_model is None:
            return [], []
        m = self.patmax_model
        accept = (_SEARCH_ACCEPT_OVERRIDE
                  if _SEARCH_ACCEPT_OVERRIDE is not None
                  else m.accept_threshold)
        # Quét góc: override angle range của model (0-0) bằng dải search.
        if _SEARCH_ANGLE_RANGE > 0:
            a_lo, a_hi, a_st = (-_SEARCH_ANGLE_RANGE, _SEARCH_ANGLE_RANGE,
                                 _SEARCH_ANGLE_STEP)
        else:
            a_lo, a_hi, a_st = m.angle_low, m.angle_high, m.angle_step
        # Search với ngưỡng thấp (_SEARCH_DIAG_FLOOR) + num_results dư để
        # bắt cả near-miss cho diagnostic. Lọc theo `accept` thật ở Python.
        floor = min(accept, _SEARCH_DIAG_FLOOR) if _SEARCH_DIAG else accept
        n_req = max(m.num_results, m.num_results + 4) if _SEARCH_DIAG else m.num_results
        results, _ = run_patmax_align(
            img_bgr, m,
            algorithm="PatQuick",
            train_mode_align=m.train_mode if m.train_mode else "Image",
            accept_threshold=floor,
            angle_low=a_lo, angle_high=a_hi, angle_step=a_st,
            scale_low=m.scale_low, scale_high=m.scale_high,
            scale_step=getattr(m, "scale_step", 0.1) or 0.1,
            num_results=n_req,
            overlap_threshold=m.overlap_threshold,
            coarse_downscale=1,
            build_score_map=False,
        )
        accepted, rejected = [], []
        for r in results:
            d = {
                "x": float(r.x), "y": float(r.y),
                "w": float(r.width), "h": float(r.height),
                "score": float(r.score),
                "angle": float(r.angle),
            }
            if r.score >= accept:
                accepted.append(d)
            elif _SEARCH_DIAG and r.score >= _SEARCH_DIAG_FLOOR:
                rejected.append(d)
        return accepted, rejected

    def AOI(self, moment1, moment2, moment3):
        self.check_sample = self.sys.find('Test_sample')
        # Capture frame (giữ logic loop 4 frame để warm sensor)
        i = 0
        while True:
            i += 1
            ret, self.frame = self.video.read()
            self.frame = cv2.resize(self.frame, (880, 680))
            if i == 4:
                cv2.imwrite(r"img/1.png", self.frame)
                break

        self.img = cv2.imread('img/1.png')
        # Mask trắng vùng giữa (motor body) — PatMax tìm screw ở 3 góc, không
        # quan tâm vùng giữa. Mask trắng = không edge → score thấp = bỏ qua.
        cv2.rectangle(self.img, (166, 175), (516, 464), (255, 255, 255), -1)

        # ── PatQuick screw detection ───────────────────────────────
        detections, rejected = self._patmax_detect_screws(self.img)
        count = len(detections)

        def _draw_box(d, color):
            x_c, y_c = d["x"], d["y"]
            w, h = d["w"], d["h"]
            x1 = int(round(x_c - w / 2)); y1 = int(round(y_c - h / 2))
            x2 = int(round(x_c + w / 2)); y2 = int(round(y_c + h / 2))
            cv2.rectangle(self.img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.img, f"{d['score']:.2f}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

        # Xanh = accepted (đếm vào count). Cam = near-miss dưới ngưỡng —
        # diagnostic: nhìn score cam để biết hạ _SEARCH_ACCEPT_OVERRIDE.
        for d in detections:
            _draw_box(d, (0, 255, 0))
        for d in rejected:
            _draw_box(d, (0, 165, 255))

        # ── Moments overlay (giữ nguyên logic cũ) ──────────────────
        if 0.4 <= moment2 <= 0.6:
            cv2.putText(self.img, str(moment2), org=(340, 195),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(0, 255, 0), thickness=3)
        else:
            cv2.putText(self.img, str(moment2), org=(340, 195),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=3)

        if 0.4 <= moment3 <= 0.6:
            cv2.putText(self.img, str(moment3), org=(107, 452),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(0, 255, 0), thickness=3)
        else:
            cv2.putText(self.img, str(moment3), org=(107, 452),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=3)

        if 0.4 <= moment1 <= 0.6:
            cv2.putText(self.img, str(moment1), org=(775, 470),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(0, 255, 0), thickness=3)
        else:
            cv2.putText(self.img, str(moment1), org=(775, 470),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=3)

        # ── OK/NG verdict ──────────────────────────────────────────
        # PASS: tìm thấy đúng 3 screws (PatMax match cùng template "screw")
        # + 3 moments trong range [0.4, 0.6]. Không cần check label như cũ
        # vì PatMax model = 1 pattern duy nhất (tất cả results đều là screw).
        ok = (count == 3
              and 0.4 <= moment1 <= 0.6
              and 0.4 <= moment2 <= 0.6
              and 0.4 <= moment3 <= 0.6)
        if ok:
            cv2.putText(self.img, "OK", org=(600, 180),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3.0,
                        color=(0, 255, 0), thickness=3)
            if self.check_sample is False:
                self.count_OK += 1
                self.count_ok.configure(text='OK: ' + str(self.count_OK))
                self.sys_quantity.replace(str(self.count_OK), 'OK')
            self.lbl_OK_NG.configure(text='OK', text_color='green')
            self.result = 1
        else:
            cv2.putText(self.img, "NG", org=(600, 180),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3.0,
                        color=(0, 0, 255), thickness=3)
            if self.check_sample is False:
                self.count_NG += 1
                self.count_ng.configure(text='NG: ' + str(self.count_NG))
                self.sys_quantity.replace(str(self.count_NG), 'NG')
            self.lbl_OK_NG.configure(text='NG', text_color='red')
            self.result = 10

        if self.check_sample is False:
            self.count_Total += 1
            self.count_total.configure(text='Total: ' + str(self.count_Total))
            self.sys.replace('False', 'Test_sample')

    def display_canvas_L(self, moment1, moment2, moment3):
        self.AOI(moment1, moment2, moment3)
        img_print = cv2.resize(self.img, (self.canvas_L.winfo_width(), self.canvas_L.winfo_height()))
        img_print = cv2.cvtColor(img_print, cv2.COLOR_RGB2BGR)
        self.photo_L = ImageTk.PhotoImage(image=Image.fromarray(img_print))
        self.canvas_L.create_image(0, 0, image=self.photo_L, anchor=tkinter.NW)

    def display_canvas_R(self, moment1, moment2, moment3):
        self.AOI(moment1, moment2, moment3)
        img_print = cv2.resize(self.img, (self.canvas_R.winfo_width(), self.canvas_R.winfo_height()))
        img_print = cv2.cvtColor(img_print, cv2.COLOR_RGB2BGR)
        self.photo_R = ImageTk.PhotoImage(image=Image.fromarray(img_print))
        self.canvas_R.create_image(0, 0, image=self.photo_R, anchor=tkinter.NW)

    def display_canvas_T(self, moment1, moment2, moment3):
        self.AOI(moment1, moment2, moment3)
        img_print = cv2.resize(self.img, (self.canvas_R.winfo_width(), self.canvas_R.winfo_height()))
        img_print = cv2.cvtColor(img_print, cv2.COLOR_RGB2BGR)
        self.photo_R = ImageTk.PhotoImage(image=Image.fromarray(img_print))
        self.canvas_R.create_image(0, 0, image=self.photo_R, anchor=tkinter.NW)

# App().mainloop()
