"""
Подготовка датасета SHAPR для multi-view 2D->3D реконструкции.

По умолчанию генерируются 4 проекции:
  - top_proj    — сумма верхней половины Z, деление на 32
  - bottom_proj — сумма нижней половины Z, деление на 32
  - side_proj   — сумма по Y, деление на 64
  - front_proj  — сумма по X, деление на 64

Использование:
  python3 src/prepare_dataset.py --shapr_dir data/raw/shapr/dataset_for_3D_reconstruction
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from morphometrics import extract_all_metrics

try:
    from src.reconstruction_utils import add_complexity_score, extract_all_views, get_view_names
except ImportError:
    from reconstruction_utils import add_complexity_score, extract_all_views, get_view_names


def parse_filename(filename: str) -> dict[str, object]:
    """Извлекает тип клетки и score из имени файла SHAPR."""
    stem = Path(filename).stem

    rbc_match = re.match(r"^(-?\d+\.\d+)_(.+?)(\d{6})$", stem)
    if rbc_match:
        return {
            "cell_type": rbc_match.group(2),
            "score": float(rbc_match.group(1)),
            "category": "RBC",
            "index": rbc_match.group(3),
        }

    ipsc_match = re.match(r"^([A-E])_(.+?)(\d{6})$", stem)
    if ipsc_match:
        return {
            "cell_type": ipsc_match.group(2),
            "score": None,
            "category": "iPSC",
            "index": ipsc_match.group(3),
        }

    return {
        "cell_type": "unknown",
        "score": None,
        "category": "unknown",
        "index": "",
    }


def normalize_image(image_2d: np.ndarray) -> np.ndarray:
    """Нормализация 2D image для справочного канала image/."""
    image = image_2d.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def write_dataset_config(output_dir: Path, input_mode: str) -> None:
    config = {
        "recommended_input_mode": input_mode,
        "generated_view_names": ["top_proj", "bottom_proj", "side_proj", "front_proj"],
        "input_view_names": list(get_view_names(input_mode)),
        "normalization": {
            "top_proj": "sum(z[:32]) / 32",
            "bottom_proj": "sum(z[32:]) / 32",
            "side_proj": "sum(axis=1) / 64",
            "front_proj": "sum(axis=2) / 64",
        },
    }
    config_path = output_dir / "dataset_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def prepare_dataset(
    shapr_dir: str,
    output_dir: str,
    rbc_only: bool = True,
    input_mode: str = "quad",
) -> None:
    """Полный пайплайн подготовки данных."""
    shapr_path = Path(shapr_dir)
    output_path = Path(output_dir)

    print("=" * 60)
    print("ПОДГОТОВКА ДАТАСЕТА")
    print("=" * 60)

    image_dir = shapr_path / "image"
    obj_dir = shapr_path / "obj"
    image_files = sorted(path.name for path in image_dir.glob("*.tif"))
    print(f"Найдено {len(image_files)} файлов в image/")

    for subdir in [
        "top_proj",
        "bottom_proj",
        "side_proj",
        "front_proj",
        "obj",
        "image",
    ]:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    skipped = 0

    for index, filename in enumerate(image_files, start=1):
        info = parse_filename(filename)
        if rbc_only and info["category"] != "RBC":
            skipped += 1
            continue

        obj_path = obj_dir / filename
        img_path = image_dir / filename
        if not obj_path.exists():
            skipped += 1
            continue

        try:
            obj_3d = tifffile.imread(obj_path)
            image_2d = tifffile.imread(img_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"  ОШИБКА {filename}: {exc}")
            skipped += 1
            continue

        if obj_3d.ndim != 3:
            skipped += 1
            continue

        obj_binary = (obj_3d > 0).astype(np.float32)
        views = extract_all_views(obj_binary)
        metrics = extract_all_metrics(obj_binary, threshold=0.5)

        safe_name = Path(filename).stem
        for view_name, projection in views.items():
            np.save(output_path / view_name / f"{safe_name}.npy", projection)

        np.save(output_path / "obj" / f"{safe_name}.npy", obj_binary)
        np.save(output_path / "image" / f"{safe_name}.npy", normalize_image(image_2d))

        record = {
            "name": safe_name,
            "cell_type": info["cell_type"],
            "category": info["category"],
            "score": info["score"],
            "label": 0 if info["cell_type"] == "discocyte" else 1,
            "obj_shape": str(obj_3d.shape),
            **metrics,
        }
        records.append(record)

        if index % 100 == 0 or index == len(image_files):
            print(f"  [{index}/{len(image_files)}] обработано")

    df = add_complexity_score(pd.DataFrame(records))
    csv_path = output_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    write_dataset_config(output_path, input_mode=input_mode)

    n_normal = int((df["label"] == 0).sum())
    n_anomaly = int((df["label"] == 1).sum())

    print(f"\n{'=' * 60}")
    print("ГОТОВО")
    print(f"  Обработано: {len(df)} сэмплов (пропущено: {skipped})")
    print(f"  Normal (discocyte): {n_normal}")
    print(f"  Anomaly: {n_anomaly}")
    print(f"  Input mode для обучения: {input_mode}")
    print("\n  По типам:")
    for cell_type, count in df["cell_type"].value_counts().sort_index().items():
        print(f"    {cell_type}: {count}")
    print(f"\n  Сохранено в: {output_path}")
    print(f"  Метаданные: {csv_path}")
    print(f"{'=' * 60}")


def main() -> None:
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
    parser.add_argument(
        "--input_mode",
        type=str,
        default="quad",
        choices=["tri", "quad"],
        help="Рекомендуемый режим входа для дальнейшего обучения",
    )
    args = parser.parse_args()

    prepare_dataset(
        args.shapr_dir,
        args.output_dir,
        rbc_only=not args.all_cells,
        input_mode=args.input_mode,
    )


if __name__ == "__main__":
    main()
