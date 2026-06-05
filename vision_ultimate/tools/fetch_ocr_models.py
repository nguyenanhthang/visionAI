#!/usr/bin/env python3
"""
fetch_ocr_models.py — Tải sẵn model cho OCR Max chạy OFFLINE (không cần mạng).

CÁCH DÙNG: chạy 1 LẦN trên máy CÓ internet, sau đó COPY cả thư mục `models/`
sang máy air-gapped (đặt cạnh app / file .exe đã đóng gói).

    python tools/fetch_ocr_models.py                       # EasyOCR (vi,en) + Tesseract (vie,eng)
    python tools/fetch_ocr_models.py --easyocr-only
    python tools/fetch_ocr_models.py --tesseract-only --tess-langs vie eng
    python tools/fetch_ocr_models.py --easyocr-langs vi en ja

File tải về:
    <project>/models/easyocr/*.pth           (craft_mlt_25k.pth + <script>_g2.pth)
    <project>/models/tessdata/*.traineddata
"""
from __future__ import annotations
import argparse
import io
import os
import ssl
import zipfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # <project> (cha của tools/)
EASYOCR_DIR = os.path.join(ROOT, "models", "easyocr")
TESSDATA_DIR = os.path.join(ROOT, "models", "tessdata")

# URL trực tiếp (fallback khi máy build KHÔNG cài easyocr). JaidedAI đóng .pth
# trong file .zip ở các release. latin_g2 phủ chữ Latin → đủ cho vie + eng.
EASYOCR_DIRECT = {
    "craft_mlt_25k.pth":
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
    "latin_g2.pth":
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip",
}
TESS_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"


def _get(url: str, timeout: int = 120) -> bytes:
    """GET nhị phân; nếu mạng công ty chặn SSL cert → thử lại bỏ verify."""
    try:
        with urllib.request.urlopen(url, context=ssl.create_default_context(),
                                    timeout=timeout) as r:
            return r.read()
    except ssl.SSLError:
        with urllib.request.urlopen(url, context=ssl._create_unverified_context(),
                                    timeout=timeout) as r:
            return r.read()


def fetch_easyocr(langs) -> None:
    os.makedirs(EASYOCR_DIR, exist_ok=True)
    # Cách 1 (chuẩn nhất): nhờ chính EasyOCR tải đúng model vào đúng chỗ.
    try:
        import easyocr  # noqa: F401
        print(f"[easyocr] Dùng thư viện easyocr tải model {list(langs)} → {EASYOCR_DIR}")
        easyocr.Reader(list(langs), gpu=False, verbose=True,
                       model_storage_directory=EASYOCR_DIR, download_enabled=True)
        print("[easyocr] ✓ Xong (qua thư viện).")
        return
    except ImportError:
        print("[easyocr] Máy này chưa cài easyocr → tải trực tiếp .zip (chỉ vi/en).")
    except Exception as e:
        print(f"[easyocr] Tải qua thư viện lỗi: {e}\n          → thử tải trực tiếp.")

    # Cách 2: tải .zip trực tiếp.
    for fname, url in EASYOCR_DIRECT.items():
        dst = os.path.join(EASYOCR_DIR, fname)
        if os.path.isfile(dst):
            print(f"[easyocr] ✓ đã có {fname}, bỏ qua.")
            continue
        print(f"[easyocr] tải {fname} ← {url}")
        try:
            with zipfile.ZipFile(io.BytesIO(_get(url))) as zf:
                member = next(n for n in zf.namelist() if n.endswith(".pth"))
                with zf.open(member) as src, open(dst, "wb") as out:
                    out.write(src.read())
            print(f"[easyocr] ✓ {fname}")
        except Exception as e:
            print(f"[easyocr] ✗ {fname}: {e}\n"
                  f"          Tải tay {url} → giải nén lấy .pth → bỏ vào {EASYOCR_DIR}")


def fetch_tesseract(langs) -> None:
    os.makedirs(TESSDATA_DIR, exist_ok=True)
    for lang in langs:
        dst = os.path.join(TESSDATA_DIR, f"{lang}.traineddata")
        if os.path.isfile(dst):
            print(f"[tessdata] ✓ đã có {lang}.traineddata, bỏ qua.")
            continue
        url = TESS_URL.format(lang=lang)
        print(f"[tessdata] tải {lang}.traineddata ← {url}")
        try:
            with open(dst, "wb") as f:
                f.write(_get(url))
            print(f"[tessdata] ✓ {lang}.traineddata")
        except Exception as e:
            print(f"[tessdata] ✗ {lang}: {e}\n          Tải tay {url} → bỏ vào {TESSDATA_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tải model OCR cho chế độ offline.")
    ap.add_argument("--easyocr-only", action="store_true", help="chỉ tải EasyOCR")
    ap.add_argument("--tesseract-only", action="store_true", help="chỉ tải Tesseract")
    ap.add_argument("--easyocr-langs", nargs="+", default=["vi", "en"],
                    help="mã ngôn ngữ EasyOCR (vd: vi en ja)")
    ap.add_argument("--tess-langs", nargs="+", default=["vie", "eng"],
                    help="mã ngôn ngữ Tesseract (vd: vie eng)")
    a = ap.parse_args()

    print(f"Project: {ROOT}")
    if not a.tesseract_only:
        fetch_easyocr(a.easyocr_langs)
    if not a.easyocr_only:
        fetch_tesseract(a.tess_langs)
    print("\nXONG. Giờ COPY cả thư mục models/ sang máy offline (đặt cạnh app/.exe):")
    print(f"  EasyOCR : {EASYOCR_DIR}")
    print(f"  Tessdata: {TESSDATA_DIR}")


if __name__ == "__main__":
    main()
