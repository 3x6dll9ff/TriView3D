---
title: Thesis Plan
date: 2026-03-23
tags: [thesis, plan, roadmap]
related: [[docs/architecture/overview]], [[docs/decisions/log]], [[docs/dataset_analysis]]
---

# Дипломная работа — Dual-View 3D Cell Shape Prediction & Classification
**Автор:** Данила Кардашевский | **Дедлайн:** 2 месяца | **Уровень:** Бакалавриат (EECS, UDG)

---

## СУТЬ РАБОТЫ

Разрабатываем пайплайн, который из **двух 2D срезов z-стека** (верхний и нижний) **предсказывает полную 3D форму клетки**, а затем **классифицирует** её морфологию (норма / аномалия).

### Почему это работает и зачем нужно
- Полный z-стек конфокальной микроскопии = 50–100 срезов → дорого и долго
- Если можно точно реконструировать 3D из всего 2 срезов → **ускорение скрининга в разы**
- SHAPR (2022) доказал: из 1 среза можно предсказать 3D. **Мы показываем: из 2 срезов — лучше, плюс добавляем классификацию**

### Уникальный вклад
- **Dual-view encoder** — 2 среза (top + bottom) вместо одного → больше информации о 3D форме
- **Classification head** на основе предсказанной 3D → ни в одной найденной работе нет связки "dual-view reconstruction + morphology classification"
- Сравнение с single-view (SHAPR) и 2D baseline → количественное доказательство преимущества

---

## ПАЙПЛАЙН (step-by-step)

### Шаг 1: Данные — SHAPR Dataset
**Что:** z-стеки клеток (RBC + iPSC nuclei) с 3D ground truth масками
**Откуда:** https://zenodo.org/records/7031924 (3.6 ГБ)
**Зачем:** У каждой клетки есть:
- `images/` — 2D срез клетки (64×64)
- `mask/` — 2D маска сегментации (64×64)
- `obj/` — **ground truth 3D форма** (64×64×64)

**Почему именно SHAPR:**
- Единственный открытый датасет с парами (2D срез → 3D ground truth)
- Авторы SHAPR доказали: из 1 среза можно предсказать 3D (мы делаем из 2)
- RBC имеют разнообразные формы: здоровые дискоциты, стоматоциты, эхиноциты → основа для классификации

### Шаг 2: Подготовка dual-view входа
**Что:** Из z-стека берём 2 среза — верхний (top) и нижний (bottom)
**Зачем:** Два среза с разных глубин несут больше информации о 3D форме, чем один
**Как:**
```
z-стек (N срезов) → top = срез[0], bottom = срез[N-1]
→ dual input: 2 × (64×64) → тензор [2, 64, 64]
```
**Почему 2 среза, а не 3–5:** Минимальное количество исходных данных → максимально практично. Показываем, что даже 2 среза дают значимое улучшение над 1.

### Шаг 3: Dual-View 3D Autoencoder (реконструкция)
**Что:** Нейросеть, которая из 2 срезов предсказывает полную 3D форму
**Зачем:** Learned reconstruction → научно обоснованный метод (vs intensity layering = ненаучный)
**Архитектура:**
```
Input:  [batch, 2, 64, 64]          ← 2 среза клетки

Encoder (2D ConvNet):
  Conv2d → BN → ReLU → Pool         ← извлечение фичей из обоих срезов
  → latent vector (128-dim)

Decoder (3D ConvTranspose):
  ConvTranspose3d → BN → ReLU       ← генерация 3D формы из latent
  → output [batch, 1, 64, 64, 64]

Loss: BCE(predicted_3d, ground_truth_3d) + Dice loss
```
**Почему autoencoder, а не MONAI DenseNet:** DenseNet = классификатор. Нам сначала нужно **реконструировать** 3D, и только потом классифицировать. Autoencoder делает именно это.

### Шаг 4: Морфологические метрики из 3D
**Что:** Из предсказанной 3D формы извлекаем количественные параметры
**Зачем:** Объективные метрики формы → основа для классификации
**Метрики:**
- **Volume** — объём объекта (сумма voxel'ей)
- **Sphericity** — насколько форма близка к сфере
- **Convexity** — отношение объёма к выпуклой оболочке
- **Surface roughness** — шероховатость поверхности
- **Eccentricity** — вытянутость

**Почему метрики, а не end-to-end:** Интерпретируемость. Можно сказать "клетка аномальная, потому что sphericity = 0.62 при норме > 0.85". На защите это сильный аргумент.

### Шаг 5: Классификация (норма vs аномалия)
**Что:** По морфометрическим метрикам классифицируем клетки
**Как создаём метки:**
- RBC в SHAPR имеют разные формы (дискоциты, стоматоциты, эхиноциты)
- Нормальные RBC = правильные дискоциты (sphericity в определённом диапазоне, стабильный convexity)
- Аномальные = отклонения от нормальной формы
- Метки создаём через **кластеризацию** ground truth 3D + экспертные пороги на метрики

**Модели:**
| Модель | Роль | Зачем |
|--------|------|-------|
| Random Forest | Baseline | Классификация по ручным метрикам |
| 3D CNN (classification head) | Основная | End-to-end на reconstructed 3D |
| 2D CNN на одном срезе | Baseline | Proof: 3D лучше чем 2D |

### Шаг 6: Визуализация и интерпретация
**Что:** 3D визуализация + Grad-CAM + SHAP
**Зачем:** "Вау-эффект" на защите + объяснимость
- Plotly 3D: predicted shape vs ground truth (overlay)
- Grad-CAM 3D: какие области 3D формы важны для классификации
- SHAP: какие метрики (sphericity, convexity...) влияют на решение

---

## НАУЧНАЯ БАЗА

| Работа | Суть | Наше отличие |
|--------|------|-------------|
| **SHAPR** (iScience 2022) | 1 срез 2D → 3D shape reconstruction | Мы: **2 среза** (dual-view) → 3D + **classification** |
| **MVN-AFM** (Nature Comms 2024) | Multi-view AFM → 3D fusion | Другой тип данных, нет classification |
| **AFM Cancer ML** (IEEE 2023) | Random Forest на height maps | Нет 3D reconstruction, нет CNN |

---

## СТЕК ТЕХНОЛОГИЙ

```
Python 3.10+
torch           — autoencoder, 3D CNN, training loop
numpy           — матрицы, морфометрия
scipy           — sphericity, convex hull, surface metrics
scikit-learn    — Random Forest baseline, метрики, кластеризация
plotly          — 3D визуализация (interactive)
matplotlib      — графики loss/accuracy/confusion matrix
shap            — интерпретируемость классификатора
opencv-python   — загрузка изображений, препроцессинг
tifffile        — чтение .tif z-стеков из SHAPR
scikit-image    — морфологические операции, marching cubes
```

---

## СТРУКТУРА РЕПОЗИТОРИЯ

```
thesis-cell-3d/
├── data/
│   ├── raw/                    # SHAPR датасет (не коммитим)
│   └── processed/              # подготовленные тензоры
├── src/
│   ├── download_data.py        # скачивание SHAPR
│   ├── prepare_dataset.py      # извлечение top/bottom, создание меток
│   ├── dataset.py              # PyTorch Dataset
│   ├── autoencoder.py          # Dual-view autoencoder (2D→3D)
│   ├── classifier.py           # Classification head + Random Forest
│   ├── train_reconstruction.py # обучение autoencoder
│   ├── train_classifier.py     # обучение классификатора
│   ├── morphometrics.py        # извлечение 3D метрик
│   ├── evaluate.py             # метрики: IoU, Dice, accuracy, AUROC
│   └── visualize.py            # Plotly 3D, Grad-CAM, SHAP
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_reconstruction.ipynb
│   ├── 03_classification.ipynb
│   └── 04_results.ipynb
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
├── .gitignore
└── THESIS_PLAN.md
```

---

## ПЛАН ПО НЕДЕЛЯМ

### Неделя 1–2: Данные + разведка
- [ ] Скачать SHAPR датасет
- [ ] Исследовать структуру: сколько клеток, формат, размеры z-стеков
- [ ] Написать `prepare_dataset.py`: извлечь top/bottom срезы, создать пары (2×2D, 3D ground truth)
- [ ] Визуализировать примеры: 2D срезы + 3D ground truth
- [ ] Рассчитать морфометрию на ground truth 3D → создать метки normal/anomaly

**Результат:** подготовленный датасет с парами (dual-view input, 3D target, label)

### Неделя 3–4: Dual-View Autoencoder
- [ ] Написать `autoencoder.py`: 2D encoder + 3D decoder
- [ ] Написать `dataset.py`: PyTorch Dataset для загрузки пар
- [ ] Написать `train_reconstruction.py`: training loop
- [ ] Обучить на SHAPR: target Dice > 0.75
- [ ] Сравнить: dual-view (2 среза) vs single-view (1 срез) → IoU, Dice

**Результат:** обученный autoencoder + метрики реконструкции

### Неделя 5–6: Классификация + метрики
- [ ] Написать `morphometrics.py`: volume, sphericity, convexity, roughness
- [ ] Написать `classifier.py`: classification head на latent + Random Forest на метриках
- [ ] Обучить и сравнить: 3D CNN vs RF vs 2D baseline
- [ ] Написать `evaluate.py`: accuracy, AUROC, F1, confusion matrix

**Результат:** сравнительная таблица методов, confusion matrix, AUROC

### Неделя 7: Визуализация + интерпретация
- [ ] Plotly 3D: predicted vs ground truth (overlay)
- [ ] Grad-CAM 3D: подсветка аномальных зон
- [ ] SHAP на морфометрических фичах
- [ ] Срезы (xy, xz, yz) для слайдов

**Результат:** интерактивные 3D визуализации + interpretability

### Неделя 8: Текст диплома
1. Введение — актуальность, цель
2. Обзор литературы — SHAPR, MVN-AFM, AFM Cancer ML + 5–7 дополнительных
3. Методика — dual-view autoencoder, архитектура, датасет, метрики
4. Результаты — таблицы, графики, 3D визуализации
5. Обсуждение — limitations, future work
6. Заключение

---

## МЕТРИКИ УСПЕХА

### Реконструкция (autoencoder)
| Метрика | Минимум | Хорошо | Отлично |
|---------|---------|--------|---------|
| Dice score | > 0.70 | > 0.80 | > 0.88 |
| IoU | > 0.55 | > 0.70 | > 0.80 |
| Dual vs single | +любой % | +3% Dice | +5%+ Dice |

### Классификация
| Метрика | Минимум | Хорошо | Отлично |
|---------|---------|--------|---------|
| Accuracy | > 75% | > 85% | > 92% |
| AUROC | > 0.80 | > 0.88 | > 0.93 |
| Vs 2D baseline | +любой % | +5% | +10%+ |

---

## МОЩНОСТИ

| Задача | Где | Время |
|--------|-----|-------|
| Подготовка данных, визуализация | MacBook (CPU) | часы |
| Обучение autoencoder | Google Colab T4 | 2–4 ч |
| Обучение классификатора | Google Colab T4 | 30–60 мин |

---

## ЧТО ГОВОРИТЬ НА ЗАЩИТЕ

> *"Мы разработали dual-view пайплайн, который из двух срезов z-стека предсказывает полную 3D форму клетки с помощью autoencoder и классифицирует морфологические аномалии. Метод показывает Dice X для реконструкции и accuracy Y% для классификации, превосходя single-view и 2D baseline. В отличие от полного z-стека (50–100 срезов), достаточно двух изображений."*

---

## ТЕКУЩИЙ СТАТУС

- [ ] Неделя 1–2: не начато
- [ ] Неделя 3–4: не начато
- [ ] Неделя 5–6: не начато
- [ ] Неделя 7: не начато
- [ ] Неделя 8: не начато
