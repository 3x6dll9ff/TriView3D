---
title: Project Structure
date: 2026-04-08
tags: [structure, architecture, reference]
related: [[docs/architecture/overview]], [[knowledge/stack]]
---

# Структура проекта

```
MultiView_3D_Prediction/
│
├── src/                            # Исходный код (Python)
│   ├── api.py                      #   FastAPI REST API
│   ├── autoencoder.py              #   Tri-View Autoencoder (2D→3D)
│   ├── classifier.py               #   MLP + Random Forest классификатор
│   ├── dataset.py                  #   PyTorch Dataset
│   ├── download_data.py            #   Скачивание SHAPR с Zenodo
│   ├── evaluate.py                 #   Метрики: Dice, IoU, accuracy, AUROC
│   ├── morphometrics.py            #   Извлечение 3D морфометрических метрик
│   ├── prepare_dataset.py          #   Подготовка tri-view пар + метки
│   ├── train_classifier.py         #   Training loop классификатора
│   ├── train_reconstruction.py     #   Training loop автоэнкодера
│   ├── visualize.py                #   Plotly 3D, срезы, сравнение
│   └── visualize_examples.py       #   Визуализация примеров датасета
│
├── frontend/                       # Web-приложение (React + Vite)
│   ├── src/                        #   React компоненты
│   ├── public/                     #   Статические ассеты
│   ├── Dockerfile.frontend         #   Docker-сборка фронтенда
│   ├── package.json                #   npm зависимости
│   └── vite.config.ts              #   Конфигурация Vite
│
├── data/                           # Данные (не в Git)
│   ├── raw/                        #   SHAPR датасет (3.6 ГБ)
│   │   └── shapr/                  #     image/, mask/, obj/
│   └── processed/                  #   Подготовленные .npy тензоры
│       ├── top_proj/               #     Верхние проекции
│       ├── bottom_proj/            #     Нижние проекции
│       ├── side_proj/              #     Боковые проекции
│       ├── obj/                    #     3D ground truth
│       ├── image/                  #     2D срезы
│       └── metadata.csv            #     Метаданные + метки
│
├── results/                        # Результаты экспериментов
│   ├── figures/                    #   Графики и визуализации
│   ├── metrics/                    #   JSON с метриками
│   └── best_autoencoder.pt         #   Лучшая модель (не в Git)
│
├── notebooks/                      # Jupyter ноутбуки
│   └── train_colab.ipynb           #   Обучение в Google Colab
│
├── docs/                           # Документация
│   ├── thesis_plan.md              #   План дипломной работы
│   ├── dataset_analysis.md         #   Анализ SHAPR датасета
│   ├── project_structure.md        #   Этот файл
│   ├── architecture/               #   Архитектура
│   │   └── overview.md             #     Схема пайплайна
│   ├── decisions/                  #   Лог архитектурных решений
│   │   └── log.md                  #     ADR записи
│   ├── api/                        #   Документация API
│   │   └── README.md               #     Эндпоинты FastAPI
│   ├── daily/                      #   Ежедневные заметки
│   ├── assets/                     #   Вложения (изображения)
│   └── _templates/                 #   Шаблоны Obsidian
│       └── note.md                 #     Базовый шаблон заметки
│
├── knowledge/                      # База знаний агента
│   ├── rules.md                    #   Правила разработки
│   ├── glossary.md                 #   Глоссарий терминов
│   └── stack.md                    #   Стек технологий
│
├── .agent/                         # AI-агент
│   ├── skills/                     #   Навыки (SKILL.md)
│   │   ├── frontend/SKILL.md       #     Frontend стандарты
│   │   ├── backend/SKILL.md        #     Backend стандарты
│   │   ├── engineering/SKILL.md    #     Инженерные стандарты
│   │   └── data-science/SKILL.md   #     ML/DS стандарты
│   ├── memory/                     #   Runtime state (не в Git)
│   └── prompts/                    #   Шаблоны промптов
│
├── .obsidian/                      # Конфигурация Obsidian
│
├── Dockerfile.backend              # Docker-сборка бэкенда
├── docker-compose.yml              # Оркестрация контейнеров
├── requirements.txt                # Python зависимости
├── .gitignore                      # Git ignore rules
├── .dockerignore                   # Docker ignore rules
└── README.md                       # Точка входа
```

## Пайплайн запуска

```bash
# 1. Скачать данные
python3 src/download_data.py

# 2. Подготовить датасет
python3 src/prepare_dataset.py

# 3. Обучить autoencoder (на Colab с GPU)
python3 src/train_reconstruction.py --epochs 50

# 4. Обучить классификатор
python3 src/train_classifier.py --mode rf
python3 src/train_classifier.py --mode latent --autoencoder results/best_autoencoder.pt

# 5. Оценка метрик
python3 src/evaluate.py

# 6. Docker (backend + frontend)
docker compose up --build
```

## Файлы src/

| Файл | Назначение | Статус |
|------|-----------|--------|
| `download_data.py` | Скачивание SHAPR с Zenodo | ✅ |
| `prepare_dataset.py` | Подготовка tri-view пар + метки | ✅ |
| `dataset.py` | PyTorch Dataset (tri-view + single-view) | ✅ |
| `autoencoder.py` | Tri-View Autoencoder (2D→3D) | ✅ |
| `classifier.py` | MLP на latent + Random Forest | ✅ |
| `morphometrics.py` | 6 морфометрических метрик из 3D | ✅ |
| `train_reconstruction.py` | Training loop для autoencoder | ✅ |
| `train_classifier.py` | Training loop для классификатора | ✅ |
| `evaluate.py` | Dice, IoU, accuracy, AUROC | ✅ |
| `visualize.py` | Plotly 3D, срезы, сравнение | ✅ |
| `visualize_examples.py` | Визуализация примеров датасета | ✅ |
| `api.py` | FastAPI REST API | ✅ |
