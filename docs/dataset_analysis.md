---
title: Dataset Analysis
date: 2026-03-23
tags: [data, shapr, analysis]
related: [[docs/thesis_plan]], [[knowledge/glossary]]
---

# SHAPR Dataset — Анализ

## Источник
- **URL:** https://zenodo.org/records/7031924
- **Размер:** 3.6 ГБ (ZIP)
- **Статья:** SHAPR — An AI approach to predict 3D cell shapes from 2D microscopy images (iScience 2022)

## Структура

Плоская структура (без поддиректорий для датасетов):
```
dataset_for_3D_reconstruction/
├── image/   — 825 файлов, 2D срезы клеток (.tif, 64×64, ~4.4 KB)
├── mask/    — 825 файлов, 2D маски сегментации (.tif, 64×64)
├── obj/     — 825 файлов, 3D ground truth формы (.tif, 64×64×64, ~273 KB)
├── Cylinder_fit/   — 825 файлов, цилиндрические фиты
├── Ellipse_fit/    — 825 файлов, эллиптические фиты
└── Results .../    — результаты из статей SHAPR
```

## Типы клеток (извлечены из имён файлов)

Имена файлов содержат **тип клетки** и **числовой score**:

| Score | Тип | Кол-во | Описание |
|-------|-----|--------|----------|
| -1.00 | `spherocyte` | 93 | Сферические, патология |
| -0.67 | `stomatocyte_II` | 11 | Глубокая впадина |
| -0.33 | `stomatocyte_I` | 30 | Лёгкая впадина |
| **0.00** | **`discocyte`** | **176** | **Нормальная форма RBC** |
| 0.33 | `echinocyte_I` | 47 | Лёгкие шипы |
| 0.67 | `echinocyte_II` | 77 | Средние шипы |
| 1.00 | `echinocyte_III` | 168 | Выраженные шипы |
| A | `cell_clusters` | 69 | iPSC кластеры |
| B | `keratocytes` | 31 | iPSC кератоциты |
| C | `knizocytes` | 23 | iPSC книзоциты |
| D | `multilobate_cells` | 12 | iPSC многолопастные |
| E | `acanthocytes` | 88 | Шиповидные |

**Итого:** 825 клеток, ~7 морфотипов RBC + 5 типов iPSC

## Как создавать метки для классификации

### Вариант 1: Бинарный (normal vs anomaly)
- **Normal (0):** `discocyte` (score = 0.00) → 176 клеток
- **Anomaly (1):** всё остальное → 649 клеток
- ⚠️ Дисбаланс 176 vs 649 — нужен class weight или oversampling

### Вариант 2: Мультиклассовый (по морфотипу)
- 12 классов по типу клетки
- Более информативно, но некоторые классы маленькие (stomatocyte_II = 11)

### Вариант 3: По score (регрессия)
- Score от -1.0 до 1.0 — степень деформации
- Нужно исключить iPSC клетки (A-E)

**Рекомендация:** Вариант 1 для основного эксперимента, Вариант 2 как дополнительный.

## Важные замечания

1. **Метки уже есть в именах файлов** — не нужен KMeans или morphometry-based labeling
2. **image/ содержит 2D срезы** (не z-стеки) — один срез на клетку
3. **obj/ содержит полный 3D** ground truth — из него извлекаем top/bottom срезы для dual-view
4. Все файлы `.tif` формата — использовать `tifffile` для загрузки
