#!/usr/bin/env python3
"""
fetch_ocr_models.py — Tải sẵn model cho OCR Max chạy OFFLINE (không cần mạng).

CÁCH DÙNG: chạy 1 LẦN trên máy CÓ internet, sau đó COPY cả thư mục `models/`
sang máy air-gapped (đặt cạnh app / file .exe đã đóng gói).

    python tools/fetch_ocr_models.py                       # EasyOCR + Tesseract
    python tools/fetch_ocr_models.py --paddle-only         # CHỈ PaddleOCR (khuyên dùng)
    python tools/fetch_ocr_models.py --paddle              # thêm PaddleOCR vào bộ mặc định
    python tools/fetch_ocr_models.py --paddle-only --paddle-lang en
    python tools/fetch_ocr_models.py --easyocr-only --easyocr-langs vi en ja

File tải về:
    <project>/models/paddle/{det,rec,cls}/   (PaddleOCR — inference model)
    <project>/models/easyocr/*.pth
    <project>/models/tessdata/*.traineddata
"""
from __future__ import annotations
import argparse
import io
import os
import shutil
import ssl
import zipfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # <project> (cha của tools/)
PADDLE_DIR = os.path.join(ROOT, "models", "paddle")
EASYOCR_DIR = os.path.join(ROOT, "models", "easyocr")
TESSDATA_DIR = os.path.join(ROOT, "models", "tessdata")

# Bộ EasyOCR cơ bản phủ vie + eng. JaidedAI đóng .pth trong .zip ở các release.
EASYOCR_DIRECT = {
    "craft_mlt_25k.pth":
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
    "latin_g2.pth":
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip",
    "english_g2.pth":
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
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


# ── PaddleOCR (khuyên dùng) ───────────────────────────────────────────
def fetch_paddle(lang: str = "vi") -> None:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("[paddle] Chưa cài. Chạy: pip install paddlepaddle paddleocr")
        return
    print(f"[paddle] Khởi tạo PaddleOCR(lang={lang}) để tải model về cache…")
    try:
        try:
            PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        except TypeError:
            PaddleOCR(lang=lang)            # PaddleOCR 3.x đổi tham số
    except Exception as e:
        print(f"[paddle] tải lỗi (kiểm tra mạng): {e}")
        return
    # Copy model det/rec/cls từ ~/.paddleocr → models/paddle/
    cache = os.path.join(os.path.expanduser("~"), ".paddleocr")
    found = {"det": None, "rec": None, "cls": None}
    for root, _dirs, files in os.walk(cache):
        if any(f.startswith("inference.pd") for f in files):
            low = root.replace("\\", "/").lower()
            for kind in found:
                if "/" + kind in low and found[kind] is None:
                    found[kind] = root
    os.makedirs(PADDLE_DIR, exist_ok=True)
    for kind, srcd in found.items():
        if not srcd:
            print(f"[paddle] ! chưa thấy model {kind} trong {cache} "
                  "(một số bản không dùng 'cls' — thiếu cls vẫn chạy được)")
            continue
        dstd = os.path.join(PADDLE_DIR, kind)
        shutil.rmtree(dstd, ignore_errors=True)
        shutil.copytree(srcd, dstd)
        print(f"[paddle] ✓ {kind} → {dstd}")
    print(f"[paddle] Xong. Model offline ở {PADDLE_DIR}")


# ── EasyOCR ───────────────────────────────────────────────────────────
def _direct_pth(fname: str, url: str) -> None:
    dst = os.path.join(EASYOCR_DIR, fname)
    if os.path.isfile(dst):
        print(f"[easyocr] ✓ đã có {fname}, bỏ qua.")
        return
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


def fetch_easyocr(langs) -> None:
    os.makedirs(EASYOCR_DIR, exist_ok=True)
    for fname, url in EASYOCR_DIRECT.items():       # bộ cơ bản phủ vie/eng
        _direct_pth(fname, url)
    exotic = [l for l in langs if l not in ("vi", "en")]
    if exotic:
        try:
            import easyocr  # noqa: F401
            print(f"[easyocr] tải model ngôn ngữ thêm {exotic} qua thư viện…")
            easyocr.Reader(list(langs), gpu=False, verbose=True,
                           model_storage_directory=EASYOCR_DIR, download_enabled=True)
            print("[easyocr] ✓ xong ngôn ngữ thêm.")
        except ImportError:
            print(f"[easyocr] Cần model cho {exotic} nhưng chưa cài easyocr. "
                  f"Tải tại https://www.jaided.ai/easyocr/modelhub/ → bỏ vào {EASYOCR_DIR}")
        except Exception as e:
            print(f"[easyocr] tải {exotic} lỗi: {e}")


# ── Tesseract ─────────────────────────────────────────────────────────
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
    ap.add_argument("--paddle", action="store_true", help="tải thêm PaddleOCR")
    ap.add_argument("--paddle-only", action="store_true", help="CHỈ tải PaddleOCR")
    ap.add_argument("--paddle-lang", default="vi", help="lang PaddleOCR (vi/en/japan…)")
    ap.add_argument("--easyocr-only", action="store_true", help="CHỈ tải EasyOCR")
    ap.add_argument("--tesseract-only", action="store_true", help="CHỈ tải Tesseract")
    ap.add_argument("--easyocr-langs", nargs="+", default=["vi", "en"])
    ap.add_argument("--tess-langs", nargs="+", default=["vie", "eng"])
    a = ap.parse_args()

    any_only = a.paddle_only or a.easyocr_only or a.tesseract_only
    do_paddle = a.paddle or a.paddle_only
    do_easy = a.easyocr_only or not any_only
    do_tess = a.tesseract_only or not any_only

    print(f"Project: {ROOT}")
    if do_paddle:
        fetch_paddle(a.paddle_lang)
    if do_easy:
        fetch_easyocr(a.easyocr_langs)
    if do_tess:
        fetch_tesseract(a.tess_langs)
    print("\nXONG. Giờ COPY cả thư mục models/ sang máy offline (đặt cạnh app/.exe):")
    if do_paddle:
        print(f"  Paddle  : {PADDLE_DIR}")
    if do_easy:
        print(f"  EasyOCR : {EASYOCR_DIR}")
    if do_tess:
        print(f"  Tessdata: {TESSDATA_DIR}")


if __name__ == "__main__":
    main()
