---
name: data-science
description: Data Science and ML senior standards — reproducibility, pipelines, model development, Colab discipline
---

# Data Science & ML — Senior Standards

Все правила ниже обязательны при работе с `src/`, `data/`, `results/`, `notebooks/`.

## Reproducibility (Воспроизводимость)

### Seeds
- Все seeds задаются **один раз** в точке входа (`main()` или начало ноутбука):
```python
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
- Seed хранится как константа, не аргумент CLI (если не нужен grid search).

### Hyperparameters
- Все гиперпараметры — через `argparse` или YAML-конфиг. **Никогда магические числа в коде.**
- Текущие дефолты проекта задокументированы:
  - `latent_dim=256`, `lr=1e-3`, `batch_size=8`, `epochs=50`
  - `bce_weight=0.5`, `dice_weight=0.5`
  - `train_ratio=0.8`, `seed=42`
- При изменении гиперпараметра — фиксировать в `results/metrics/` с указанием коммита.

### Experiment Logging
- Каждый запуск обучения сохраняет в `results/metrics/`:
  - JSON с полной историей (`train_loss`, `test_loss`, `test_dice`, `test_iou`)
  - Git commit hash (добавлять через `subprocess.check_output(["git", "rev-parse", "HEAD"])`)
  - Все аргументы CLI
- Контрольная проверка: если `results/metrics/` уже содержит историю, предупреждать перед перезаписью.

## Data Pipeline

### Принципы
- `data/raw/` — **священное хранилище**: никогда не модифицировать, только читать.
- `data/processed/` — производное от `data/raw/`, полностью воспроизводится через `python3 src/prepare_dataset.py`.
- Не коммитить данные в Git — только скрипты, которые их создают.

### Валидация данных
- При загрузке датасета проверять:
```python
assert top.shape == (64, 64), f"Expected (64,64), got {top.shape}"
assert top.dtype == np.float32, f"Expected float32, got {top.dtype}"
assert 0 <= top.min() and top.max() <= 1.0, "Values out of [0, 1] range"
```
- `metadata.csv` проверяется на старте: нет NaN, все файлы существуют, все метки валидны.

### Split Strategy
- Split фиксирован через `seed=42` в `CellTriViewDataset`.
- **Не менять seed между экспериментами** — иначе тестовый набор пересечётся с тренировочным.
- Соотношение `80/20` (train/test), стратификация по `label` при дисбалансе.

## Model Development

### Порядок работы
1. **Baseline первый**: `SingleViewAutoencoder` (1 проекция) — всегда готов до экспериментов с tri-view.
2. **Один параметр за раз**: при изменении `latent_dim` — всё остальное фиксировано.
3. **Валидация до обучения**: определить метрику (Dice), порог успеха (>0.75), стратегию early stopping.

### Метрики (не accuracy!)
- **Реконструкция**: Dice, IoU, MSE — все три обязательны.
- **Классификация**: Accuracy, AUROC, F1, Confusion Matrix — accuracy одна недостаточна при дисбалансе.
- Per-class метрики обязательны: отдельно для normal и anomaly.
- Loss curves (train + test) сохраняются как JSON и визуализируются.

### Чекпоинты
- Best model по `dice` — `results/best_autoencoder.pt` (только `state_dict`).
- Full checkpoint каждые 10 эпох: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_dice`.
- **Никогда не сохранять только в конце** — сессия может умереть.

## Tensor Shape Discipline

### Обязательные аннотации
Каждая операция с тензорами документируется комментарием формы:
```python
# tri_input: (B, 3, 64, 64) — top + bottom + side
# target_3d: (B, 1, 64, 64, 64) — ground truth voxels
# latent: (B, 256) — encoded representation
# pred: (B, 1, 64, 64, 64) — predicted voxels
```

### Правила
- `unsqueeze(0)` / `squeeze(0)` всегда с комментарием зачем.
- При `view()` / `reshape()` указывать исходную и целевую форму.
- Переход 2D→3D (encoder→decoder) — ключевое место, всегда приводить размерности:
```python
# z: (B, 256) → (B, 256, 4, 4, 4) для 3D ConvTranspose
x = x.view(-1, 256, 4, 4, 4)
```

## Code Standards

### Структура скрипта
```
docstring модуля → imports → constants → classes/functions → main() → if __name__
```

### Python-специфика
- `pathlib.Path` вместо `os.path.join` — в новом коде обязательно, в существующем менять при касании.
- `logging` вместо `print()` — в production-коде (`api.py`). В скриптах обучения допускается `print()` для прогресса.
- Type hints на каждой сигнатуре: аргументы + возвращаемое значение.
- Никогда голый `except:` — ловить конкретные исключения (`FileNotFoundError`, `ValueError`).
- f-strings для форматирования, не `.format()` и не `%`.

### Импорты
Порядок: stdlib → third-party → local. Разделять пустой строкой:
```python
import argparse
import json

import numpy as np
import torch

from autoencoder import TriViewAutoencoder
```

## Google Colab Discipline

### Структура ноутбука (`notebooks/train_colab.ipynb`)
1. Mount Drive → clone репо → pip install
2. Распаковка `data_processed.zip` с Drive
3. Обучение с `--epochs 50`
4. Копирование `results/` обратно на Drive

### Правила
- `drive.mount()` — первая ячейка, всегда.
- Чекпоинт каждые 10 эпох на Drive (не только в `/content/`).
- Логирование в файл: `tee results/training.log`.
- Скрипт обучения **идемпотентный**: умеет возобновлять из checkpoint.
- **Не хранить данные в `/content/`** без бэкапа — сессия умрёт.

## Визуализация

### Правила
- Графики сохраняются в `results/figures/` как PNG (300 dpi для диплома).
- Plotly 3D — для интерактивных демо, matplotlib — для статичных графиков.
- Каждый график имеет: заголовок, подписи осей, легенду, единицы измерения.
- Шрифт на графиках: 12pt минимум (для читаемости в дипломе).

---

## Class Imbalance (КРИТИЧНО для этого проекта)

Проблема: 176 discocyte (normal) vs 426 аномалий (только RBC) — соотношение ~1:2.4.

### Что делать
```python
# 1. class_weight в loss function
from torch.nn import BCEWithLogitsLoss
pos_weight = torch.tensor([n_normal / n_anomaly])  # ~0.41
criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

# 2. WeightedRandomSampler в DataLoader
from torch.utils.data import WeightedRandomSampler
weights = [1.0 / class_count[label] for label in labels]
sampler = WeightedRandomSampler(weights, len(weights))
loader = DataLoader(dataset, sampler=sampler)  # НЕ shuffle=True!

# 3. Для Random Forest — class_weight="balanced"
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
```
### Чего НЕ делать
- Не использовать accuracy как основную метрику при дисбалансе — baseline 71% просто предсказывая всех "anomaly".
- Не делать oversampling до balanced без проверки на overfitting.
- Не игнорировать per-class метрики — F1 для каждого класса отдельно.

## GPU Memory Management

### OOM Prevention
```python
# Перед обучением — проверить доступную память
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU memory: {gpu_mem:.1f} GB")
    # T4 = 16 GB, batch_size=8 для 3D volumes — ~4 GB
    # Если мало памяти — уменьшить batch_size

# При OOM — graceful handling
try:
    pred = model(inputs)
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        # Halve batch size and retry, or skip batch
        logger.warning("OOM! Clearing cache. Consider reducing batch_size.")
        raise
    raise
```

### Mixed Precision (ускорение обучения на T4 в 1.5-2x)
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for batch in loader:
    optimizer.zero_grad()
    with autocast(device_type="cuda"):
        pred = model(inputs)
        loss = reconstruction_loss(pred, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
**Trade-off**: Mixed precision даёт 1.5-2x ускорение на T4, но может незначительно повлиять на precision метрик. Для 3D binary volumes (наш случай) — влияние минимальное.

### Gradient Clipping
```python
# Обязателен при exploding gradients (3D ConvTranspose склонен к ним)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Data Augmentation для 3D

### Допустимые трансформации
```python
# Для 2D проекций (input):
# ✅ Random horizontal/vertical flip — клетки симметричны
# ✅ Random rotation 90° — биологически инвариантно
# ✅ Gaussian noise (σ=0.01-0.05) — аугментация на шум микроскопа
# ✅ Intensity scaling ±10%

# ❌ Random crop — 64×64 и так маленькое
# ❌ Color jitter — одноканальные изображения
# ❌ Perspective transform — нет биологического обоснования
```

### Для 3D ground truth
```python
# Если flip/rotate вход — ОБЯЗАТЕЛЬНО применить ту же трансформацию к 3D target!
# Иначе модель учится на несогласованных парах.
if random.random() > 0.5:
    tri_input = np.flip(tri_input, axis=2)  # flip по X
    target_3d = np.flip(target_3d, axis=3)  # тот же flip в 3D!
```

## Overfitting Detection

### Сигналы
| Сигнал | Порог | Действие |
|--------|-------|----------|
| train_loss падает, test_loss растёт | >5 эпох подряд | Early stopping |
| train_dice > test_dice на >0.1 | Любой момент | Добавить dropout / augmentation |
| test_dice стагнирует | >15 эпох | Reduce LR или остановить |

### Early Stopping
```python
patience = 10
epochs_no_improve = 0

for epoch in range(epochs):
    ...
    if test_dice > best_dice:
        best_dice = test_dice
        epochs_no_improve = 0
        save_best_model()
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

## Statistical Significance

### Когда заявлять "tri-view лучше single-view"
```python
# Недостаточно: "tri-view Dice = 0.82, single-view = 0.79"
# Необходимо: paired t-test на per-sample Dice scores

from scipy import stats

# Dice для каждой клетки, не средний
triview_dices = [dice(pred_i, gt_i) for pred_i, gt_i in tri_pairs]
singleview_dices = [dice(pred_i, gt_i) for pred_i, gt_i in single_pairs]

t_stat, p_value = stats.ttest_rel(triview_dices, singleview_dices)
# p < 0.05 → статистически значимо
# p < 0.01 → сильно значимо
# p >= 0.05 → разница может быть случайной → нельзя заявлять в дипломе
```

### Confidence Intervals
```python
import numpy as np
dices = np.array(triview_dices)
mean = dices.mean()
std = dices.std()
ci_95 = 1.96 * std / np.sqrt(len(dices))
# Репортить: "Dice = 0.82 ± 0.03 (95% CI)"
```
**На защите:** если разница tri-view vs single-view 3%, а CI ±5% — преподаватель может атаковать. Проверяй **до** написания текста.

## Edge Cases в Morphometrics

### Degenerate Shapes
```python
def compute_sphericity(volume: np.ndarray) -> float:
    """Sphericity с защитой от edge cases."""
    voxel_count = volume.sum()
    
    # Edge case: пустой volume
    if voxel_count < 10:
        return 0.0  # Не NaN, не exception
    
    # Edge case: marching_cubes может упасть на flat volumes
    try:
        verts, faces, _, _ = marching_cubes(volume, level=0.5)
    except (ValueError, RuntimeError):
        return 0.0
    
    # Edge case: слишком мало вершин для вычислений
    if len(verts) < 4:
        return 0.0
    
    surface_area = mesh_surface_area(verts, faces)
    if surface_area < 1e-8:
        return 0.0
    
    # Формула sphericity
    return (np.pi ** (1/3) * (6 * voxel_count) ** (2/3)) / surface_area
```

### Convex Hull Failure
```python
from scipy.spatial import ConvexHull, QhullError

try:
    hull = ConvexHull(points)
except QhullError:
    # Случается при coplanar points (плоская клетка)
    return 1.0  # Assume convex if can't compute
```

## Learning Rate Strategy

### Текущая: ReduceLROnPlateau
```python
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
```
**Trade-off**: простой и надёжный, но не оптимальный.

### Рекомендация: Cosine Annealing + Warmup
```python
# Warmup предотвращает нестабильность на первых эпохах
# (особенно важен для 3D ConvTranspose — большие градиенты)
warmup_epochs = 5
for epoch in range(epochs):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = base_lr * 0.5 * (1 + cos(pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
```

## Model Versioning

### Naming Convention
```
results/
├── best_autoencoder.pt                    # Текущая лучшая (symlink/copy)
├── triview_v1_dice0.82_2026-04-08.pt      # Версионированные модели
├── triview_v2_augmented_dice0.85.pt
└── singleview_baseline_dice0.79.pt
```
**Зачем**: через 2 месяца ты не вспомнишь какой .pt файл давал какие метрики. Название = полная информация.

