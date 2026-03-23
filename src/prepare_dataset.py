"""
Подготовка датасета из SHAPR для tri-view обучения.

Tri-view = 3 ортогональные Sum Projection (карты толщины):
  - Top proj:   sum по Z-оси (axis=0) → вид сверху [H, W]
  - Front proj: sum по Y-оси (axis=1) → вид спереди [D, W]
  - Side proj:  sum по X-оси (axis=2) → вид сбоку [D, H]

Пайплайн:
  1. Сканирует image/ и obj/ в SHAPR
  2. Извлекает тип клетки из имени файла → метка normal/anomaly
  3. Из 3D ground truth (obj/) создаёт 3 Sum Projection → tri-view вход
  4. Сохраняет в data/processed/

Использование:
  python3 src/prepare_dataset.py --shapr_dir data/raw/shapr/dataset_for_3D_reconstruction
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import tifffile

from morphometrics import extract_all_metrics

# Маппинг типов клеток → категория
CELL_TYPE_MAP = {
    "spherocyte": ("RBC", -1.00),
    "stomatocyte_II": ("RBC", -0.67),
    "stomatocyte_I": ("RBC", -0.33),
    "discocyte": ("RBC", 0.00),
    "echinocyte_I": ("RBC", 0.33),
    "echinocyte_II": ("RBC", 0.67),
    "echinocyte_III": ("RBC", 1.00),
    "cell_clusters": ("iPSC", None),
    "keratocytes": ("iPSC", None),
    "knizocytes": ("iPSC", None),
    "multilobate_cells": ("iPSC", None),
    "acanthocytes": ("iPSC", None),
}


def parse_filename(filename: str) -> dict:
    """Извлекает тип клетки и score из имени файла SHAPR.

    Примеры:
      -0.33_stomatocyte_I000000.tif → type=stomatocyte_I, score=-0.33
      0.00_discocyte000136.tif → type=discocyte, score=0.00
      A_cell_clusters000604.tif → type=cell_clusters, score=None
    """
    stem = os.path.splitext(filename)[0]

    rbc_match = re.match(r"^(-?\d+\.\d+)_(.+?)(\d{6})$", stem)
    if rbc_match:
        score = float(rbc_match.group(1))
        cell_type = rbc_match.group(2)
        idx = rbc_match.group(3)
        return {
            "cell_type": cell_type,
            "score": score,
            "category": "RBC",
            "index": idx,
        }

    ipsc_match = re.match(r"^([A-E])_(.+?)(\d{6})$", stem)
    if ipsc_match:
        cell_type = ipsc_match.group(2)
        idx = ipsc_match.group(3)
        return {
            "cell_type": cell_type,
            "score": None,
            "category": "iPSC",
            "index": idx,
        }

    return {"cell_type": "unknown", "score": None, "category": "unknown", "index": ""}


def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Нормализация в [0, 1]."""
    s = s.astype(np.float32)
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 1e-6:
        return (s - s_min) / (s_max - s_min)
    return np.zeros_like(s)


def extract_tri_view(
    volume_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Из 3D volume создаёт 3 проекции: сверху, снизу, сбоку.

    Top    = sum верхней половины z-стека → вид сверху [H, W]
    Bottom = sum нижней половины z-стека → вид снизу [H, W]
    Side   = sum по Y-оси (axis=1) → боковой профиль [D, W]

    Top и Bottom различаются для асимметричных клеток (discocyte, stomatocyte).
    Side показывает профиль клетки — для discocyte = гантель.
    """
    binary = (volume_3d > 0).astype(np.float32)
    z_mid = binary.shape[0] // 2

    # Вид сверху: sum верхней половины z-стека (индексы от 0 до z_mid)
    top_proj = binary[:z_mid].sum(axis=0)     # [H, W]

    # Вид снизу: sum нижней половины z-стека (индексы от z_mid до конца)
    bottom_proj = binary[z_mid:].sum(axis=0)  # [H, W]

    # Боковой профиль: sum по Y
    side_proj = binary.sum(axis=1)            # [D, W]

    return normalize_slice(top_proj), normalize_slice(bottom_proj), normalize_slice(side_proj)


def prepare_dataset(
    shapr_dir: str,
    output_dir: str,
    rbc_only: bool = True,
) -> None:
    """Полный пайплайн подготовки данных."""
    print("=" * 60)
    print("ПОДГОТОВКА ДАТАСЕТА")
    print("=" * 60)

    image_dir = os.path.join(shapr_dir, "image")
    obj_dir = os.path.join(shapr_dir, "obj")

    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".tif"))
    print(f"Найдено {len(image_files)} файлов в image/")

    # Создаём директории
    for subdir in ["top_proj", "bottom_proj", "side_proj", "obj", "image"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    records = []
    skipped = 0

    for i, fname in enumerate(image_files):
        info = parse_filename(fname)

        if rbc_only and info["category"] != "RBC":
            skipped += 1
            continue

        obj_path = os.path.join(obj_dir, fname)
        img_path = os.path.join(image_dir, fname)

        if not os.path.exists(obj_path):
            skipped += 1
            continue

        try:
            obj_3d = tifffile.imread(obj_path)
            image_2d = tifffile.imread(img_path)

            if obj_3d.ndim != 3:
                skipped += 1
                continue

            # Tri-view: top/bottom/side проекции
            top_proj, bottom_proj, side_proj = extract_tri_view(obj_3d)

            # Бинаризация 3D
            obj_binary = (obj_3d > 0).astype(np.float32)

            # Морфометрия
            metrics = extract_all_metrics(obj_binary, threshold=0.5)

            # Метка
            is_normal = info["cell_type"] == "discocyte"
            label = 0 if is_normal else 1

            # Сохранение
            safe_name = os.path.splitext(fname)[0]
            np.save(os.path.join(output_dir, "top_proj", f"{safe_name}.npy"), top_proj)
            np.save(os.path.join(output_dir, "bottom_proj", f"{safe_name}.npy"), bottom_proj)
            np.save(os.path.join(output_dir, "side_proj", f"{safe_name}.npy"), side_proj)
            np.save(os.path.join(output_dir, "obj", f"{safe_name}.npy"), obj_binary)
            np.save(
                os.path.join(output_dir, "image", f"{safe_name}.npy"),
                normalize_slice(image_2d.astype(np.float32)),
            )

            record = {
                "name": safe_name,
                "cell_type": info["cell_type"],
                "category": info["category"],
                "score": info["score"],
                "label": label,
                "obj_shape": str(obj_3d.shape),
                **metrics,
            }
            records.append(record)

        except Exception as e:
            print(f"  ОШИБКА {fname}: {e}")
            skipped += 1
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"  [{i + 1}/{len(image_files)}] обработано")

    # Сохранение метаданных
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)

    n_normal = (df["label"] == 0).sum()
    n_anomaly = (df["label"] == 1).sum()

    print(f"\n{'=' * 60}")
    print("ГОТОВО")
    print(f"  Обработано: {len(df)} сэмплов (пропущено: {skipped})")
    print(f"  Normal (discocyte): {n_normal}")
    print(f"  Anomaly: {n_anomaly}")
    print(f"\n  По типам:")
    for ct in df["cell_type"].unique():
        n = (df["cell_type"] == ct).sum()
        print(f"    {ct}: {n}")
    print(f"\n  Сохранено в: {output_dir}")
    print(f"  Метаданные: {csv_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка SHAPR датасета")
    parser.add_argument(
        "--shapr_dir",
        type=str,
        default="data/raw/shapr/dataset_for_3D_reconstruction",
        help="Путь к распакованному SHAPR",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Директория для результатов",
    )
    parser.add_argument(
        "--all_cells",
        action="store_true",
        help="Включить iPSC клетки (по умолчанию только RBC)",
    )
    args = parser.parse_args()
    prepare_dataset(args.shapr_dir, args.output_dir, rbc_only=not args.all_cells)


if __name__ == "__main__":
    main()
