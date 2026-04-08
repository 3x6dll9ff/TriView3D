# MultiView 3D Cell Shape Prediction & Classification

Пайплайн предсказания 3D-формы клетки из трёх 2D-проекций (top, bottom, side) 
с последующей морфологической классификацией (норма / аномалия).

## Суть проекта

- **Вход:** 3 проекции клетки `[3, 64, 64]` (top + bottom + side sum projections)
- **Выход:** предсказанная 3D-форма `[1, 64, 64, 64]` + классификация
- **Данные:** [SHAPR Dataset](https://zenodo.org/records/7031924) (825 клеток, 3.6 ГБ)
- **Научная база:** расширение подхода [SHAPR (iScience 2022)](https://doi.org/10.1016/j.isci.2022.104523) — tri-view вместо single-view + classification head

## Быстрый старт

### Локально (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Скачать и подготовить данные
python3 src/download_data.py
python3 src/prepare_dataset.py

# Обучение (GPU рекомендуется)
python3 src/train_reconstruction.py --epochs 50
```

### Docker (backend + frontend)

```bash
docker compose up --build
```

После запуска:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Health-check: `http://localhost:8000/health`

```bash
docker compose down
```

### Google Colab

Используй `notebooks/train_colab.ipynb` — ноутбук автоматически клонирует репозиторий, 
устанавливает зависимости и запускает обучение на GPU T4.

## Структура проекта

Подробная структура: [`docs/project_structure.md`](docs/project_structure.md)

```
src/              — исходный код (Python)
frontend/         — web-приложение (React + Vite)
data/             — данные (не в Git)
results/          — результаты экспериментов
notebooks/        — Jupyter ноутбуки
docs/             — документация
knowledge/        — база знаний
```

## Документация

| Документ | Описание |
|----------|----------|
| [`docs/thesis_plan.md`](docs/thesis_plan.md) | План дипломной работы |
| [`docs/architecture/overview.md`](docs/architecture/overview.md) | Архитектура пайплайна |
| [`docs/decisions/log.md`](docs/decisions/log.md) | Лог ключевых решений (ADR) |
| [`docs/dataset_analysis.md`](docs/dataset_analysis.md) | Анализ SHAPR датасета |

## Стек

Python 3.10+ · PyTorch · FastAPI · React · Vite · Docker

Полный список: [`knowledge/stack.md`](knowledge/stack.md)
