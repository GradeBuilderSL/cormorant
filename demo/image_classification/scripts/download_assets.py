#!/usr/bin/env python3
"""
download_assets.py — fetch ONNX + ImageNet labels and preprocess the user's
input images for the KV260 image-classification demo.

Performs three things, each idempotent and individually skippable:
  1. Downloads ONNX models from the Google Drive folder via `gdown --folder`.
  2. Downloads the ImageNet 1000-class index JSON, then derives the 1001-class
     label list MobileNetV1 expects (class 0 = "background").
  3. Walks `assets/images/`, opens each JPG/PNG with Pillow, resizes to
     `preprocess.input_size` (default 224), normalises per
     `preprocess.normalize`, and packs every image into a single binary
     blob `assets/preprocessed/images.bin` plus a tab-separated
     `manifest.txt` (one row per image: name<TAB>byte_offset).

The encoded element type matches the deployed IP: `ap_fixed<16,8>`.  Each
image becomes `3 * H * W * 2` bytes in NCHW order.

Usage:
  python3 scripts/download_assets.py
  python3 scripts/download_assets.py --force         # rebuild everything
  python3 scripts/download_assets.py --skip-models   # only labels + preprocess
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Tuple

DEMO_DIR     = Path(__file__).resolve().parent.parent
ASSETS_DIR   = DEMO_DIR / "assets"
IMAGES_DIR   = ASSETS_DIR / "images"
LABELS_DIR   = ASSETS_DIR / "labels"
MODELS_DIR   = ASSETS_DIR / "models"
PREP_DIR     = ASSETS_DIR / "preprocessed"

LABELS_FILE  = LABELS_DIR / "imagenet_1001_labels.txt"
IMAGES_BIN   = PREP_DIR / "images.bin"
MANIFEST     = PREP_DIR / "manifest.txt"

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _http_download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    _log(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "axi-demo/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=64 * 1024)
    tmp.rename(dst)


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — copy "
            f"image_classification_config.json.example and edit it."
        )
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX models (Google Drive)
# ──────────────────────────────────────────────────────────────────────────────

def _gdown_folder(url: str, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"  gdown --folder {url}")
    cmd = [sys.executable, "-m", "gdown", "--folder", url,
           "--output", str(out_dir), "--quiet"]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(
            f"gdown failed (rc={rc}). Install with: pip install gdown\n"
            f"If the folder is rate-limited, run gdown manually and place "
            f"the .onnx files into {out_dir}/"
        )
    return sorted(p for p in out_dir.rglob("*.onnx") if p.is_file())


def download_models(folder_url: str, want: Iterable[str],
                    force: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"ONNX models → {MODELS_DIR}")

    want    = list(want)
    have    = {p.name: p for p in MODELS_DIR.glob("*.onnx")}
    missing = [n for n in want if n not in have]

    if missing or force:
        staging = MODELS_DIR / "_drive"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            for f in _gdown_folder(folder_url, staging):
                dst = MODELS_DIR / f.name
                if dst.exists() and not force:
                    continue
                shutil.copy2(f, dst)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    have = {p.name: p for p in MODELS_DIR.glob("*.onnx")}
    out  = {}
    for n in want:
        if n not in have:
            raise RuntimeError(
                f"model '{n}' not found in {MODELS_DIR} after download. "
                f"Check the Drive folder contents or override the file name "
                f"in image_classification_config.json → models[].drive_filename."
            )
        out[n] = have[n]
        _log(f"  ✓ {n}  ({have[n].stat().st_size:,} B)")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet 1001-class labels
# ──────────────────────────────────────────────────────────────────────────────

def download_labels(url: str, force: bool = False) -> Path:
    """
    Fetch the standard 1000-class index and emit the 1001-line label file
    that MobileNetV1 expects.  Class 0 is the synthetic 'background'
    category that TF Slim prepended; classes 1..1000 are the canonical
    ImageNet validation classes (in WordNet ID order).
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if LABELS_FILE.exists() and not force:
        n = sum(1 for _ in LABELS_FILE.open())
        _log(f"  cached {LABELS_FILE.name} ({n} lines)")
        return LABELS_FILE

    raw = LABELS_DIR / "imagenet_class_index.json"
    if not raw.exists() or force:
        _http_download(url, raw)
    idx = json.loads(raw.read_text())

    # The JSON is a dict keyed by stringified ints "0".."999" mapping to
    # [synset_id, human_readable_name].  We want a flat list ordered by key.
    labels = ["background"]
    for i in range(1000):
        entry = idx.get(str(i))
        if not entry or len(entry) < 2:
            raise RuntimeError(f"imagenet_class_index.json: missing entry {i}")
        labels.append(entry[1])
    if len(labels) != 1001:
        raise RuntimeError(f"derived {len(labels)} labels, expected 1001")

    LABELS_FILE.write_text("\n".join(labels) + "\n")
    _log(f"  ✓ {LABELS_FILE.name} (1001 labels)")
    return LABELS_FILE


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_RESIZE_MODES = {
    "nearest":  "NEAREST",
    "bilinear": "BILINEAR",
    "bicubic":  "BICUBIC",
    "lanczos":  "LANCZOS",
}


def _list_input_images() -> List[Path]:
    if not IMAGES_DIR.is_dir():
        return []
    return sorted(p for p in IMAGES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _sniff_kind(path: Path) -> str:
    """Best-effort guess at a file's real type from its first bytes.  Used to
    produce a helpful error when Pillow rejects a 'JPEG' that's actually a
    saved-as-image HTML error page or similar download accident."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return "unreadable"
    if head.startswith(b"\xff\xd8\xff"):                    return "JPEG"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):               return "PNG"
    if head[:6] in (b"GIF87a", b"GIF89a"):                  return "GIF"
    if head.startswith(b"BM"):                              return "BMP"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":       return "WEBP"
    if head[:2] in (b"II", b"MM"):                          return "TIFF?"
    stripped = head.lstrip().lower()
    if stripped.startswith((b"<!doctype", b"<html", b"<?xml", b"<svg")):
        return "HTML/XML"
    if all(0x09 <= b <= 0x7e or b in (0x0a, 0x0d) for b in head):
        return "text"
    return "unknown"


def _encode_pixels(arr, normalize: str):
    """
    Convert a float32 HxWx3 RGB array (values in [0, 255]) into the int16
    bit pattern of `ap_fixed<16,8>` in NCHW order.  Returns a contiguous
    numpy array of dtype int16, shape (3*H*W,).
    """
    import numpy as np
    a = arr.astype(np.float32)
    if   normalize == "tf":   a = (a - 127.5) / 127.5    # [-1, 1]
    elif normalize == "unit": a = a / 255.0              # [0, 1]
    elif normalize == "none": a = a / 256.0              # ≈ raw byte / 256
    else:
        raise ValueError(f"unknown normalize mode: {normalize!r} "
                         f"(expected 'tf', 'unit', 'none')")
    a = np.transpose(a, (2, 0, 1))                       # HWC → CHW
    bits = np.rint(a * 256.0)                            # ap_fixed<16,8>
    bits = np.clip(bits, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(bits.reshape(-1))


def preprocess_images(input_size: int, normalize: str,
                      resize_mode: str) -> Tuple[List[str], int]:
    """
    Walk assets/images/, encode each image into images.bin in order, and
    write manifest.txt with `<name><TAB><offset_bytes>` lines.

    Returns (image_names, bytes_per_image).
    """
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt")
    import numpy as np

    pil_mode = _RESIZE_MODES.get(resize_mode.lower())
    if pil_mode is None:
        raise ValueError(f"unknown resize mode: {resize_mode!r} "
                         f"(expected one of {sorted(_RESIZE_MODES)})")
    resample = getattr(Image, pil_mode)

    files = _list_input_images()
    if not files:
        raise RuntimeError(
            f"no images found in {IMAGES_DIR}. Drop one or more JPG/PNG "
            f"files into that directory and re-run."
        )

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    bytes_per_image = 3 * input_size * input_size * 2   # int16, NCHW
    _log(f"preprocess → {IMAGES_BIN}  "
         f"({len(files)} candidate images, {bytes_per_image:,} B each, "
         f"mode={normalize})")

    written: List[Tuple[Path, int]] = []   # (path, offset)
    skipped: List[Tuple[Path, str]] = []   # (path, reason)

    with open(IMAGES_BIN, "wb") as out:
        for path in files:
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB").resize(
                        (input_size, input_size), resample=resample)
                    arr = np.asarray(im)            # HxWx3 uint8
            except Exception as exc:
                kind = _sniff_kind(path)
                if kind == "HTML/XML":
                    reason = ("looks like HTML, not an image — the download "
                              "probably saved an error page; replace the file")
                elif kind == "text":
                    reason = ("looks like a text file — wrong extension or "
                              "broken download")
                elif kind == "unreadable":
                    reason = f"could not read the file ({exc})"
                else:
                    reason = f"Pillow rejected it (detected as {kind}; {exc})"
                _log(f"  ⚠ skipping {path.name}: {reason}")
                skipped.append((path, reason))
                continue

            buf = _encode_pixels(arr, normalize).tobytes()
            if len(buf) != bytes_per_image:
                raise RuntimeError(
                    f"{path.name}: encoded {len(buf)} B, expected {bytes_per_image}")
            written.append((path, out.tell()))
            out.write(buf)

    if not written:
        # Wipe the empty bin so a stale 0-byte file isn't left behind.
        try:
            IMAGES_BIN.unlink()
        except OSError:
            pass
        details = "\n".join(f"  - {p.name}: {r}" for p, r in skipped)
        raise RuntimeError(
            f"no usable images in {IMAGES_DIR} — every candidate failed:\n"
            f"{details}\nDrop at least one valid JPG/PNG and re-run.")

    with open(MANIFEST, "w") as f:
        for path, off in written:
            f.write(f"{path.name}\t{off}\n")

    for path, _ in written:
        _log(f"  ✓ {path.name}")
    if skipped:
        _log(f"  {len(skipped)} file(s) skipped — see warnings above")
    return [p.name for p, _ in written], bytes_per_image


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _preflight(cfg: dict, *, need_models: bool, need_labels: bool,
               need_preprocess: bool) -> bool:
    ok = True
    if need_models:
        try:
            import gdown    # noqa: F401
        except ImportError:
            _log("error: 'gdown' not installed — pip install -r requirements.txt")
            ok = False
        if not cfg.get("drive", {}).get("folder_url"):
            _log("error: drive.folder_url missing in config")
            ok = False
        if not cfg.get("models"):
            _log("warning: no models listed in config; nothing to download")
    if need_labels and not cfg.get("labels", {}).get("url"):
        _log("error: labels.url missing in config")
        ok = False
    if need_preprocess:
        try:
            import PIL      # noqa: F401
            import numpy    # noqa: F401
        except ImportError as exc:
            _log(f"error: missing dependency for preprocessing — {exc}")
            ok = False
        if not _list_input_images():
            _log(f"warning: no JPG/PNG files in {IMAGES_DIR} — drop a few before "
                 f"running deploy_and_run.py")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",
                   default=str(DEMO_DIR / "image_classification_config.json"))
    p.add_argument("--force", action="store_true",
                   help="re-download cached files and rebuild images.bin")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--skip-labels", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--check-only", action="store_true",
                   help="validate config + tools and exit without downloading")
    args = p.parse_args(argv)

    cfg = _load_config(Path(args.config))
    if not _preflight(cfg,
                       need_models=not args.skip_models,
                       need_labels=not args.skip_labels,
                       need_preprocess=not args.skip_preprocess):
        return 1
    if args.check_only:
        _log("preflight: ok")
        return 0

    if not args.skip_models:
        models = cfg.get("models", [])
        want   = [m["drive_filename"] for m in models]
        url    = cfg.get("drive", {}).get("folder_url")
        if want and url:
            download_models(url, want, force=args.force)

    if not args.skip_labels:
        url = cfg.get("labels", {}).get("url")
        if url:
            download_labels(url, force=args.force)

    if not args.skip_preprocess:
        prep   = cfg.get("preprocess", {})
        size   = int(prep.get("input_size", 224))
        norm   = prep.get("normalize", "tf")
        resize = prep.get("resize", "bilinear")
        # Force re-encode whenever the user asked for --force or the
        # bin/manifest are missing.  Cheap enough to always re-run, since
        # users rarely have hundreds of images locally.
        if args.force or not IMAGES_BIN.exists() or not MANIFEST.exists():
            preprocess_images(size, norm, resize)
        else:
            # Skip re-encode if the manifest still matches the on-disk image set.
            cached = MANIFEST.read_text().splitlines()
            cached_names = [ln.split("\t", 1)[0] for ln in cached if ln.strip()]
            current = [p.name for p in _list_input_images()]
            if cached_names == current:
                _log(f"  cached {IMAGES_BIN.name} ({len(current)} images)")
            else:
                preprocess_images(size, norm, resize)

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
