# TriView3D — Полная документация пайплайна

## 1. Суть проекта

**Цель:** Из 3 ортогональных 2D проекций клетки (top, bottom, side) восстановить её полную 3D форму и классифицировать морфологию (normal vs anomaly).

**Почему это важно:** В микроскопии обычно получают 2D срезы/проекции. Полная 3D форма клетки даёт гораздо больше информации — объём, сферичность, выпуклость, шероховатость поверхности. Это критично для обнаружения аномалий (раковые клетки имеют другую морфологию).

**Аналогия:** Как CT-сканер восстанавливает 3D тело из серии 2D рентгеновских снимков — мы восстанавливаем 3D клетку из 3 проекций.

---

## 2. Данные

### 2.1 Что на входе

Для каждой клетки есть:
- **3 проекции** (64×64 пикселя, grayscale):
  - `top_proj` — вид сверху (сумма верхней половины по оси Z)
  - `bottom_proj` — вид снизу (сумма нижней половины по оси Z)
  - `side_proj` — вид сбоку (сумма по оси Y)
- **3D ground truth** (64×64×64 voxel volume) — для обучения
- **Метаданные** — `metadata.csv` с полями: name, label, cell_type, volume, sphericity, convexity, eccentricity, surface_roughness, complexity_score

### 2.2 Количество

- 602 клетки всего
- 481 train / 121 test (80/20 stratified split)
- Клетки двух типов: normal и anomaly

### 2.3 Аугментация (только для train)

| Аугментация | Что делает | Зачем |
|-------------|-----------|-------|
| `RandomHFlip2D3D` | Горизонтальный flip проекций + соответствующий flip 3D volume | Увеличивает разнообразие форм |
| `GaussianNoise` | Шум σ=0.03 на проекции | Устойчивость к шуму в данных |
| `BrightnessJitter` | Δяркость ±0.1, контраст ×[0.9, 1.1] | Устойчивость к освещению |

---

## 3. Пайплайн (4 этапа)

```
2D проекции ──→ [1. Base CNN] ──→ coarse 3D ──→ [2. Refiner] ──→ refined 3D
                                    │                                    │
                                    └──→ [3. VAE] ──→ generated 3D        │
                                                                         ▼
                                                              [4. Classifier] ──→ normal / anomaly
```

### 3.1 Этап 1 — Base CNN Autoencoder

**Файл:** `src/train_reconstruction.py` → обучает `TriViewAutoencoder`

**Простое объяснение:**
> Нейросеть, которая берёт 3 картинки (вид сверху, снизу, сбоку), "сжимает" их в компактное описание (256 чисел), а потом из этого описания "разжимает" обратно в 3D объём. Ключевая фишка — skip connections, которые подсказывают декодеру где примерно должна быть клетка.

**Архитектура:**

```
Вход: [batch, 3, 64, 64] — 3 проекции
         │
    ┌────▼────┐
    │ Encoder2D│  Conv2d layers (3→32→64→128→256) + FC → latent_dim=256
    └────┬─────┘
         │  z = [batch, 256]  — латентный вектор
         │
    ┌────▼─────────────────────────────────────────┐
    │ Decoder3D + Skip Connections                 │
    │                                                │
    │   lifted views (3 канала → 3D volume)         │
    │        │                                       │
    │   skip0: Conv3d(3→256) → downsample → ADD     │  ← 4×4×4
    │   deconv0: ConvTranspose3d(256→128)            │
    │   skip1: Conv3d(3→128) → downsample → ADD     │  ← 8×8×8
    │   deconv1: ConvTranspose3d(128→64)             │
    │   skip2: Conv3d(3→64) → downsample → ADD      │  ← 16×16×16
    │   deconv2: ConvTranspose3d(64→32)              │
    │   skip3: Conv3d(3→32) → downsample → ADD       │  ← 32×32×32
    │   deconv3: ConvTranspose3d(32→1)               │
    └────────────────────────────────────────────────┘
         │
Выход: [batch, 1, 64, 64, 64] — logits (перед sigmoid)
```

**Что такое skip connections (простое объяснение):**
> Lifted views — это когда мы каждую 2D проекцию "вытягиваем" в 3D: если проекция говорит "здесь есть материал", мы размазываем это по всей глубине. Это даёт грубое 3D приближение. Skip connections передают это грубое приближение в декодер на каждом уровне — как подсказку "примерная форма клетки вот такая, дорисуй детали".

**Skip connection формула:**
```
x_decoder[level] = x_decoder[level] + Conv3d(downsample(lifted_views))
```
Это additive residual — мы не заменяем, а добавляем.

**Параметры:** 8,444,449 (8.4M)

**Функция потерь (reconstruction_loss):**

| Компонент | Вес | Что измеряет | Простое объяснение |
|-----------|-----|--------------|-------------------|
| BCE (voxel) | 0.35 | Попиксельное совпадение | Каждый воксель правильно определён? |
| Dice loss | 0.25 | Перекрытие форм | Насколько формы похожи? |
| Projection consistency | 0.5 | Совпадение проекций | Если спроецировать 3D обратно — получим те же 2D? |
| Boundary-aware BCE | 0.15 | Качество на границах | Граница клетки нарисована точно? |

**Projection consistency (простое объяснение):**
> Самый важный вид контроля качества: если из нашей 3D модели получить проекции (вид сверху/снизу/сбоку), они должны совпадать с исходными фотографиями. Это гарантирует, что 3D форма согласована с входными данными.

**Обучение:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- LR warmup: 5 эпох (linear ramp от 0.1×lr до lr)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=4)
- Early stopping: patience=15 эпох
- Batch size: 8, Epochs: 80
- Mixed precision (AMP) на GPU
- Complexity-weighted sampling: сложные клетки чаще попадают в батч

**Результат:** Best Dice ~0.94+

---

### 3.2 Этап 2 — Detail Refiner

**Файл:** `src/train_refiner.py` → обучает `DetailRefiner`

**Простое объяснение:**
> Refiner — это "корректор". Базовая модель уже даёт хорошую форму, но может ошибаться в деталях. Refiner смотрит: где базовая модель не уверена (вероятность ~0.5 вместо 0 или 1) — там он вносит поправку. Где модель уверена — не трогает.

**Архитектура:**

```
Вход: coarse_prob + lifted_views + uncertainty
      (1 канал)  (3 канала)    (1 канал) = 5 каналов
         │
    ┌────▼────┐
    │  Stem    │  Conv3d(5→32) + BN + ReLU
    ├────┬────┤
    │ Block 1  │  ResidualBlock3D(32) — 3×3 conv + skip connection
    │ Block 2  │  ResidualBlock3D(32)
    │ Block 3  │  ResidualBlock3D(32)
    ├────┬────┤
    │  Head    │  Conv3d(32→16) + Conv3d(16→1)
    └────┬────┘
         │  residual_logits = [batch, 1, 64, 64, 64]
         │
    gating = uncertainty × 1.25   ← где неуверенность 0, gating = 0 (не трогаем)
    refined_logits = coarse_logits + gating × residual_logits
```

**Uncertainty gating (простое объяснение):**
> Uncertainty = `1 - |2 × probability - 1|`. Если probability=0 или 1 — модель уверена, uncertainty=0, gating=0 — refiner ничего не меняет. Если probability=0.5 — модель не знает, uncertainty=1, gating=1.25 — refiner вносит максимальную поправку.

**Параметры:** ~743K (очень лёгкий)

**Обучение:**
- Base model заморожена (не обучается)
- Optimizer: AdamW (lr=3e-4)
- Epochs: 50, Early stopping: patience=10

**Результат:** Best Dice **0.9701** (улучшение с ~0.94 до 0.97)

---

### 3.3 Этап 3 — Conditional VAE

**Файл:** `src/train_vae.py` → обучает `TriViewCVAE`

**Простое объяснение:**
> VAE — это "генератор". В отличие от обычного autoencoder, который всегда даёт один и тот же ответ, VAE может породить несколько вариантов 3D формы. Мы просим его сгенерировать 8 вариантов, проецируем каждый обратно в 2D и выбираем тот, чьи проекции лучше совпадают с исходными. Это best-of-K подход.

**Архитектура:**

```
Вход: [batch, 3, 64, 64]
         │
    ┌────▼────┐
    │ Encoder  │  Conv2d layers → flatten → fc_mu + fc_logvar
    └────┬─────┘
         │
    mu = [batch, 256]     ← среднее латентного распределения
    logvar = [batch, 256] ← логарифм дисперсии
         │
    ┌────▼──────────┐
    │ Reparameterize │  z = mu + exp(0.5 × logvar) × ε    (ε ~ N(0,1))
    └────┬───────────┘
         │  z = [batch, 256]  — сэмплированный латентный вектор
         │
    ┌────▼─────────────────────┐
    │ Decoder3D + Skip Conn.   │  (тот же декодер что и у CNN)
    └────┬─────────────────────┘
         │
Выход: [batch, 1, 64, 64, 64] — logits
```

**Reparameterization trick (простое объяснение):**
> Вместо того чтобы напрямую использовать mu как латентный вектор, мы добавляем случайный шум: z = mu + шум. Размер шума контролируется logvar. Если logvar маленький — VAE генерирует почти одинаковые варианты. Если большой — варианты сильно отличаются. Это позволяет исследовать пространство возможных 3D форм.

**Best-of-K inference:**

```
1. Кодируем вход → mu, logvar
2. Генерируем K=8 кандидатов:
   - Кандидат 0: z = mu (детерминированный)
   - Кандидаты 1-7: z = mu + noise (стохастические)
3. Для каждого кандидата:
   - Делаем 3D объём → проектируем обратно в 2D
   - Считаем L1 расстояние между проекциями и входом
4. Выбираем кандидата с минимальным расстоянием
```

**Функция потерь (vae_loss):**

| Компонент | Вес | Что |
|-----------|-----|-----|
| Reconstruction loss | 1.0 | Та же composite что и у CNN (BCE+Dice+Proj+Boundary) |
| KL divergence | 0.001 | Регуляризация — не даёт латентному пространству "схлопнуться" |

**KL divergence (простое объяснение):**
> Это штраф за то, что латентное распределение отличается от стандартного нормального. Без него VAE просто выучил бы детерминированное отображение (как обычный autoencoder) и генерация не работала бы. Маленький вес 0.001 — приоритет на качество реконструкции.

**Параметры:** 9,493,281 (9.5M)

**Результат:** Best Dice **0.9601** (с best-of-K=8)

---

### 3.4 Этап 4 — Classifier

**Файл:** `src/train_classifier.py`

Два режима:

#### 4a. Random Forest (baseline)

**Простое объяснение:**
> Классический ML метод, который решает задачу "по 5 числам (объём, сферичность, выпуклость, эксцентриситет, шероховатость) определи — клетка нормальная или аномальная?"

**Результат:** Accuracy 93.4%, AUROC 96.7%

**Feature importance:**
| Признак | Важность |
|---------|----------|
| eccentricity | 35.7% |
| convexity | 29.2% |
| sphericity | 18.4% |
| surface_roughness | 13.6% |
| volume | 3.1% |

#### 4b. MLP Classifier (основной)

**Простое объяснение:**
> Нейросеть, которая берёт "сжатое описание" клетки от autoencoder (256 чисел) + 5 морфометрических признаков из 3D формы → определяет normal/anomaly.

**Архитектура:**
```
[batch, 261]  (256 latent + 5 morphometrics)
    │
Linear(261→128) + ReLU + Dropout(0.3)
Linear(128→64) + ReLU + Dropout(0.2)
Linear(64→2)
    │
[batch, 2] — logits для normal/anomaly
```

---

## 4. Инференс (как работает при использовании)

### 4.1 CNN + Refiner (детерминированный путь)

```python
# 1. Загружаем 3 проекции
inputs = load_input_views(filename, view_names)  # [1, 3, 64, 64]

# 2. Base model + TTA (test-time augmentation)
coarse_logits = model(inputs)                          # прямой проход
flipped_logits = model(inputs.flip(-1)).flip(-1)         # flip + unflip
coarse_logits = (coarse_logits + flipped_logits) / 2.0   # усредняем

# 3. Refiner (если есть)
lifted = lift_views_to_volume(inputs, view_names)
refined_logits = refiner(coarse_logits, lifted)

# 4. Финальный объём
volume = torch.sigmoid(refined_logits)  # [1, 1, 64, 64, 64]
```

**TTA (простое объяснение):**
> Прогоняем модель дважды: на оригинальном входе и на зеркально отражённом. Усредняем результаты. Это бесплатно даёт +1-2% Dice — модель видит клетку с двух сторон и усредняет ошибки.

### 4.2 VAE (генеративный путь)

```python
# 1. Сэмплируем K=8 кандидатов
pred_logits, best_score = best_of_k_generate(vae, inputs, view_names, num_samples=8)

# 2. Финальный объём
volume = torch.sigmoid(pred_logits)  # [1, 1, 64, 64, 64]
```

---

## 5. Вспомогательные функции

### 5.1 Projection / Lifting

**`project_volume_batch(volume, view_names)`** — 3D → 2D
> Суммирует воксели по нужной оси. top_proj: сумма верхней половины по Z, side_proj: сумма по Y.

**`lift_views_to_volume(views, view_names)`** — 2D → 3D
> Каждую проекцию "вытягивает" в 3D: top_proj разматывается по Z, side_proj — по Y. Конкатенирует все в 3-канальный 3D volume.

### 5.2 Complexity Score

> Комплексность = z(шероховатость) + z(площадь_поверхности/объём) + z(1 - выпуклость). Используется для weighted sampling — сложные клетки чаще попадают в батч, чтобы модель лучше училась на трудных случаях.

---

## 6. Метрики

| Метрика | Что измеряет | Идеальное значение |
|---------|-------------|-------------------|
| **Dice** | Перекрытие форм (2×intersection / (pred+gt)) | 1.0 |
| **IoU** | Intersection over Union | 1.0 |
| **ASSD** | Average Symmetric Surface Distance | 0.0 |
| **HD95** | 95th percentile Hausdorff Distance | 0.0 |
| **Volume Diff** | Разница объёмов (%) | 0.0 |
| **Reprojection L1** | Разница между исходными и обратными проекциями | 0.0 |

---

## 7. Результаты обучения

| Модель | Best Dice | Параметры |
|--------|-----------|-----------|
| Base CNN (tri, skip connections) | 0.94+ | 8.4M |
| + Refiner | **0.9701** | +0.7M |
| VAE (best-of-K=8) | **0.9601** | 9.5M |
| RF Classifier | 93.4% acc, 96.7% AUROC | — |

---

## 8. Структура файлов

```
TriView3D/
├── src/
│   ├── autoencoder.py          # Encoder2D, Decoder3D, TriViewAutoencoder, все loss-функции
│   ├── vae.py                   # TriViewCVAE, best_of_k_generate, vae_loss
│   ├── refiner.py               # DetailRefiner с uncertainty gating
│   ├── classifier.py            # LatentClassifier (MLP), MorphometryRFClassifier (RF)
│   ├── dataset.py               # CellTriViewDataset + аугментации (flip, noise, brightness)
│   ├── reconstruction_utils.py  # project/lift views, complexity score, infer functions
│   ├── train_reconstruction.py  # Обучение base CNN (этап 1)
│   ├── train_refiner.py         # Обучение refiner (этап 2)
│   ├── train_vae.py             # Обучение VAE (этап 3)
│   ├── train_classifier.py      # Обучение классификатора (этап 4)
│   ├── evaluate.py              # Оценка с TTA, hard subset, per-cell-type
│   ├── api.py                   # FastAPI сервер (CNN + Refiner + VAE инференс)
│   └── prepare_dataset.py       # Генерация quad-view проекций из 3D
├── notebooks/
│   ├── train_colab.ipynb        # Этап 1 — Colab
│   ├── train_refiner_colab.ipynb# Этап 2 — Colab
│   ├── train_vae_colab.ipynb    # Этап 3 — Colab
│   └── train_classifier_colab.ipynb # Этап 4 — Colab
├── frontend/
│   └── src/App.tsx              # React UI с 3D rendering + overlay + metrics
├── results/
│   ├── best_autoencoder.pt      # Обученная base модель
│   ├── best_refiner.pt          # Обученный refiner
│   ├── best_vae.pt              # Обученный VAE
│   └── metrics/                 # JSON с историей обучения
└── scripts/
    └── upload_to_hf.py          # Загрузка моделей на HuggingFace
```

---

## 9. Ключевые технические решения

| Решение | Почему |
|---------|--------|
| Skip connections от lifted views | Pix2Vox-style — даёт +4-5% Dice, почти бесплатно (+40K params) |
| TTA (flip + average) | Бесплатный +1-2% Dice на инференсе |
| Composite loss (4 компонента) | Каждый аспект качества контролируется отдельно |
| Projection consistency via BCE | Сильнее чем L1 — заставляет модель точно воспроизводить проекции |
| Boundary-aware BCE | Граница клетки — самое важное, там вес ×2 |
| Uncertainty gating (min=0) | Refiner не портит уверенные предсказания |
| Best-of-K VAE | Генеративная неопределённость разрешается через reprojection error |
| AdamW + warmup | Стабильное обучение, weight decay для регуляризации |
| Complexity-weighted sampling | Сложные клетки не игнорируются |
| Stratified split | Баланс классов в train/test |
