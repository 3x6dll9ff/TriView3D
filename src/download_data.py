"""
Скачивание SHAPR датасета с Zenodo.

SHAPR содержит:
  - images/ — 2D срезы клеток (64×64)
  - mask/  — 2D маски сегментации
  - obj/   — 3D ground truth формы (64×64×64)

Использование:
  python3 src/download_data.py
"""

import os
import shutil
import urllib.request
import zipfile

SHAPR_ZIP_URL = (
    "https://zenodo.org/records/7031924/files/"
    "Datasets_for_3D_reconstruction.zip?download=1"
)
SHAPR_DESC_URL = (
    "https://zenodo.org/records/7031924/files/"
    "Dataset_description.pdf?download=1"
)

RAW_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "shapr_dataset.zip")
DESC_PATH = os.path.join(RAW_DIR, "Dataset_description.pdf")


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  [{pct:5.1f}%] {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)


def download_shapr() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)

    # Скачиваем описание датасета
    if not os.path.exists(DESC_PATH):
        print("Скачиваю Dataset_description.pdf...")
        urllib.request.urlretrieve(SHAPR_DESC_URL, DESC_PATH, _progress_hook)
        print("\n  Готово")
    else:
        print(f"Dataset_description.pdf уже скачан: {DESC_PATH}")

    # Скачиваем основной архив
    if not os.path.exists(ZIP_PATH):
        print(f"\nСкачиваю SHAPR датасет (~3.6 ГБ)...")
        print(f"  URL: {SHAPR_ZIP_URL}")
        urllib.request.urlretrieve(SHAPR_ZIP_URL, ZIP_PATH, _progress_hook)
        print("\n  Готово")
    else:
        size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
        print(f"ZIP уже скачан: {ZIP_PATH} ({size_mb:.0f} MB)")

    # Распаковка
    extract_dir = os.path.join(RAW_DIR, "shapr")
    if not os.path.exists(extract_dir):
        print(f"\nРаспаковываю в {extract_dir}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(extract_dir)
        print("  Готово")
    else:
        print(f"Уже распаковано: {extract_dir}")

    # Итог
    print("\n=== SHAPR датасет готов ===")
    _print_tree(extract_dir, max_depth=2)


def _print_tree(path: str, prefix: str = "", max_depth: int = 3, depth: int = 0) -> None:
    if depth >= max_depth:
        return
    entries = sorted(os.listdir(path))
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

    for f in files[:5]:
        size = os.path.getsize(os.path.join(path, f))
        print(f"{prefix}{f} ({size / 1024:.0f} KB)")
    if len(files) > 5:
        print(f"{prefix}... и ещё {len(files) - 5} файлов")

    for d in dirs:
        subpath = os.path.join(path, d)
        n_children = len(os.listdir(subpath))
        print(f"{prefix}{d}/ ({n_children} элементов)")
        _print_tree(subpath, prefix + "  ", max_depth, depth + 1)


if __name__ == "__main__":
    download_shapr()
