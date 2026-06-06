# Model OCR offline — chạy OCR Max KHÔNG cần mạng

Tool **OCR Max** mặc định chạy **100% offline**. Chỉ cần đặt sẵn file model vào
đúng thư mục dưới đây (làm 1 lần), app sẽ không bao giờ gọi internet khi Run.

```
models/
├── paddle/     # PaddleOCR (KHUYÊN DÙNG) — subfolder det/ rec/ cls/
│   ├── det/
│   ├── rec/
│   └── cls/
├── tessdata/   # *.traineddata cho Tesseract
└── easyocr/    # *.pth cho EasyOCR
```

App tự dò model theo thứ tự ưu tiên:
1. Đường dẫn nhập trong tool (*PaddleOCR / Tessdata / EasyOCR models folder*)
2. `<app>/models/{paddle,tessdata,easyocr}` (thư mục này)
3. Biến môi trường `TESSDATA_PREFIX` / `EASYOCR_MODULE_PATH`
4. `~/.EasyOCR/model`, `~/.paddleocr` (nơi engine tự lưu sau lần tải đầu)

> **Nhanh nhất:** trên máy CÓ mạng chạy `python tools/fetch_ocr_models.py --paddle-only`
> (hoặc bỏ cờ để tải cả EasyOCR+Tesseract), rồi copy nguyên thư mục `models/` sang máy offline.

---

## 0. PaddleOCR — engine KHUYÊN DÙNG (chính xác cao, tiếng Việt tốt)

**Cài thư viện:**
```bash
pip install paddlepaddle      # bản CPU (GPU: paddlepaddle-gpu)
pip install paddleocr
```

**Lấy model offline (làm 1 lần trên máy có mạng):**
```bash
python tools/fetch_ocr_models.py --paddle-only             # chỉ Paddle, lang vi
python tools/fetch_ocr_models.py --paddle-only --paddle-lang en
```
Lệnh gọi PaddleOCR tự tải model rồi copy vào `models/paddle/{det,rec,cls}`.
Sau đó copy thư mục `models/` sang máy offline.

> Thủ công: chạy app 1 lần khi CÓ mạng (Engine = paddle) → model về
> `~/.paddleocr/whl/`. Copy 3 folder con (det / rec / cls — folder chứa
> `inference.pdmodel`) vào `models/paddle/det`, `/rec`, `/cls`.

**Dùng:** node OCR Max → **Engine = `paddle`**, **Language = `vie`**. Đã đặt model
local → chạy offline. Thiếu model + không mạng → báo lỗi chỉ rõ thư mục cần đặt.

---

## 1. Tesseract (engine `tesseract` / `auto`)

**Cần:** binary Tesseract + file `*.traineddata`.

- Cài binary: Windows = UB-Mannheim installer · Ubuntu = `apt install tesseract-ocr`
  · macOS = `brew install tesseract`. Nếu không nằm trong PATH, trỏ đường dẫn ở
  param **Tesseract .exe**.
- Tải `*.traineddata` (offline, 1 lần) từ
  <https://github.com/tesseract-ocr/tessdata_fast> rồi copy vào `models/tessdata/`:

```
models/tessdata/
├── eng.traineddata
└── vie.traineddata        # tiếng Việt (param Language = "vie")
```

> Có sẵn tessdata cục bộ thì khỏi cần cài langpack hệ thống — app tự thêm
> `--tessdata-dir models/tessdata`.

## 2. EasyOCR (engine `easyocr` / `auto` fallback)

**Cần:** `pip install easyocr` + 2 loại file `.pth`. Mặc định tool **TẮT tải mạng**
(`allow_download = False`) nên phải đặt sẵn model:

```
models/easyocr/
├── craft_mlt_25k.pth      # detection — LUÔN cần (dùng chung mọi ngôn ngữ)
├── latin_g2.pth           # recognition khi Language = "vie" hoặc "vie+eng"
└── english_g2.pth         # recognition khi Language = "eng" (chỉ tiếng Anh)
```

> ⚠️ Tên file recognition **phụ thuộc Language đã chọn** trên node: `vie`→`latin_g2.pth`,
> `eng`→`english_g2.pth`, `jpn`→`japanese_g2.pth`… Nếu báo thiếu file, thông báo
> lỗi sẽ nói đúng tên cần. Tiếng Việt nên để **Language = `vie`** (latin_g2 đọc
> được cả ký tự Latin/tiếng Anh).

Lấy file `.pth` bằng 1 trong 2 cách:
- **Trên máy có mạng:** bật param **Cho phép tải model** rồi Run 1 lần → EasyOCR
  tải về `~/.EasyOCR/model/`. Copy 2 file `.pth` đó sang `models/easyocr/`.
- **Tải tay:** từ <https://www.jaided.ai/easyocr/modelhub/>.

> Ngôn ngữ khác → recognition model khác (vd `japanese_g2.pth`, `korean_g2.pth`,
> `zh_sim_g2.pth`). Detection `craft_mlt_25k.pth` luôn cần.

---

## Đóng gói (PyInstaller)

`VisionPro.spec` đã bundle cả thư mục `models/` vào app. File `.pth`/`.traineddata`
đặt ở đây trước khi build sẽ đi kèm trong `.exe` → máy đích chạy offline ngay,
không cần cài Tesseract/tải EasyOCR.
