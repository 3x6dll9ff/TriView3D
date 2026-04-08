---
title: Project Rules
date: 2026-04-08
tags: [rules, agent, workflow, standards]
related: [[knowledge/stack]], [[knowledge/glossary]], [[docs/decisions/log]]
---

# Project Rules

Обязательные правила для AI-агента и разработчика. Этот файл — живой документ, обновляется после каждой задачи.

---

## 1. Workflow агента

### Перед ЛЮБОЙ задачей (обязательно)

```
1. Определить тип задачи → выбрать скиллы из таблицы ниже
2. Прочитать SKILL.md каждого релевантного скилла через view_file
3. Прочитать knowledge/rules.md (этот файл)
4. Прочитать knowledge/mistakes.md (если существует) — не повторять ошибки
5. Проверить docs/decisions/log.md — не предлагать отвергнутые подходы
```

### Маппинг задача → скилл

| Тип задачи | Скилл(ы) |
|-----------|----------|
| Модели, обучение, датасеты, Colab | `data-science/SKILL.md` + `engineering/SKILL.md` |
| Python-код, API, Docker | `backend/SKILL.md` + `engineering/SKILL.md` |
| React, TypeScript, 3D-визуализация | `frontend/SKILL.md` + `engineering/SKILL.md` |
| Git, документация, архитектура | `engineering/SKILL.md` |
| Полный пайплайн | **Все 4 скилла** |

### После ЛЮБОЙ задачи (обязательно)

```
1. Обновить knowledge/rules.md — если выявлено новое правило
2. Обновить docs/decisions/log.md — если принято архитектурное решение
3. Если ошибка → зафиксировать в knowledge/mistakes.md
4. Если изменена структура → обновить docs/project_structure.md
```

---

## 2. Отвергнутые подходы (НЕ предлагать!)

Подходы, которые были рассмотрены и отвергнуты с обоснованием. Полная информация в [[docs/decisions/log]].

| Подход | Почему отклонён | ADR # |
|--------|----------------|-------|
| Intensity-based layering (2D→3D) | Нет физического обоснования | #1 |
| Синтетические данные (эллипсоиды) | Не валидирует на реальных клетках | #2 |
| MONAI DenseNet | Это классификатор, не реконструктор | #3 |
| KMeans для создания меток | Метки уже есть в именах файлов SHAPR | #4 |
| Binary MIP для проекций | Теряет информацию о толщине | #7 |

---

## 3. Правила кода (Python)

### Обязательно
- Type hints на **каждой** функции (аргументы + return type)
- `pathlib.Path` вместо `os.path` в новом коде; мигрировать при касании старого
- `logging` вместо `print()` в production-коде (`api.py`); `print()` допустим в скриптах обучения
- Только конкретные `except` — никогда голый `except:`
- f-strings для форматирования
- Docstring на каждой публичной функции/классе

### Tensor Shape Discipline
```python
# Каждая операция с тензорами — комментарий формы
# input: (B, 3, 64, 64) — tri-view проекции
# latent: (B, 256) — encoder output
# pred: (B, 1, 64, 64, 64) — predicted voxel grid
```

### Seeds
Все seeds в одном месте, в начале `main()`:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

### Import Order
```python
# stdlib
import argparse
import json

# third-party
import numpy as np
import torch

# local
from autoencoder import TriViewAutoencoder
```

---

## 4. Правила данных

| Правило | Детали |
|---------|--------|
| `data/raw/` священно | Никогда не модифицировать, только читать |
| `data/processed/` воспроизводимо | Пересоздаётся через `python3 src/prepare_dataset.py` |
| Split фиксирован | `seed=42`, `train_ratio=0.8`. Не менять между экспериментами |
| Валидация на входе | `assert shape == (64, 64)`, `assert dtype == float32` |

---

## 5. Правила модели

| Правило | Детали |
|---------|--------|
| Baseline first | `SingleViewAutoencoder` → затем `TriViewAutoencoder` |
| Один параметр за раз | При ablation — менять одну переменную |
| Best model по Dice | Сохраняется в `results/best_autoencoder.pt` (только `state_dict`) |
| Checkpoint каждые 10 эпох | Full checkpoint: model + optimizer + epoch + best_dice |
| Метрики полные | Dice + IoU + MSE для реконструкции; Accuracy + AUROC + F1 для классификации |

---

## 6. Правила API

| Правило | Детали |
|---------|--------|
| CORS development | `allow_origins=["*"]` — только для dev |
| Модель singleton | Загружается один раз при startup, не на каждый запрос |
| Ошибки generic | Не отдавать клиенту `str(e)`, stack trace, пути файлов |
| Inference mode | `model.eval()` + `@torch.no_grad()` — всегда |

---

## 7. Правила документации

| Правило | Детали |
|---------|--------|
| YAML frontmatter | `title`, `date`, `tags`, `related` — на каждом `.md` |
| Wikilinks | `[[docs/architecture/overview]]` для связей между файлами |
| Новые заметки | `docs/daily/` — ежедневные логи |
| Вложения | `docs/assets/` — изображения и файлы |
| Графики | `results/figures/` — PNG 300 dpi для диплома |

---

## 8. Правила Git

```
feat(scope): description     — новая функциональность
fix(scope): description      — исправление бага
refactor(scope): description — рефакторинг без изменения поведения
docs(scope): description     — только документация
test(scope): description     — тесты
chore(scope): description    — обслуживание (deps, configs)
```

Scopes: `model`, `api`, `data`, `frontend`, `docker`, `docs`

---

## 9. Известные технические долги

| Долг | Приоритет | Файл |
|------|-----------|------|
| `api.py` использует `os.path` вместо `pathlib` | Средний | `src/api.py` |
| `api.py` использует `print()` вместо `logging` | Средний | `src/api.py` |
| CORS wildcard `*` в production | Высокий | `src/api.py` |
| `App.tsx` монолит (~700 строк) — разбить на компоненты | Средний | `frontend/src/App.tsx` |
| Нет unit-тестов | Высокий | — |
| Нет path traversal protection | Высокий | `src/api.py` |
