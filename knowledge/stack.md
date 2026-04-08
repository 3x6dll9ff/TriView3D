---
title: Tech Stack
date: 2026-04-08
tags: [stack, tools, dependencies]
related: [[knowledge/glossary]], [[docs/architecture/overview]]
---

# Стек технологий

## Backend (Python)

| Пакет | Назначение | Версия |
|-------|-----------|--------|
| `torch` | Автоэнкодер, 3D CNN, training loop | ≥ 2.0.0 |
| `torchvision` | Трансформации, утилиты | ≥ 0.15.0 |
| `numpy` | Матрицы, морфометрия | ≥ 1.24.0 |
| `scipy` | Sphericity, convex hull, surface metrics | ≥ 1.11.0 |
| `scikit-learn` | Random Forest baseline, метрики, кластеризация | ≥ 1.3.0 |
| `scikit-image` | Морфологические операции, marching cubes | ≥ 0.22.0 |
| `opencv-python` | Загрузка изображений, препроцессинг | ≥ 4.8.0 |
| `tifffile` | Чтение .tif z-стеков из SHAPR | ≥ 2023.7.0 |
| `pandas` | Работа с metadata.csv | ≥ 2.1.0 |

## Визуализация

| Пакет | Назначение |
|-------|-----------|
| `plotly` | Интерактивная 3D-визуализация |
| `matplotlib` | Графики loss/accuracy/confusion matrix |
| `shap` | Интерпретируемость классификатора |

## Web-приложение

| Пакет | Назначение |
|-------|-----------|
| `fastapi` | REST API для модели |
| `uvicorn` | ASGI-сервер |
| `python-multipart` | Загрузка файлов через API |
| `streamlit` | Интерактивный дашборд для защиты |

## Frontend

| Инструмент | Версия/назначение |
|-----------|------------------|
| React + TypeScript | UI-фреймворк |
| Vite | Сборщик и dev-сервер |
| TailwindCSS | Стилизация |

## Инфраструктура

| Инструмент | Назначение |
|-----------|-----------|
| Docker + Docker Compose | Контейнеризация backend + frontend |
| Google Colab (T4 GPU) | Обучение моделей |
| Google Drive | Хранение данных и чекпоинтов |

## Версии Python
- Минимум: **Python 3.10+**
- Docker-образ: `python:3.11-slim`
