---
title: AI Agent Configuration
date: 2026-04-08
tags: [agent, config, skills]
related: [[knowledge/rules]]
---

# AI Agent

Конфигурация и правила поведения AI-агента в проекте.

## Структура

```
.agent/
├── skills/                     # Навыки (исполняемые стандарты)
│   ├── frontend/SKILL.md       #   React, TypeScript, Three.js, CSS
│   ├── backend/SKILL.md        #   Python, FastAPI, Docker, Security
│   ├── engineering/SKILL.md    #   Git, ADR, Testing, Documentation
│   └── data-science/SKILL.md   #   ML, Reproducibility, Colab, Tensors
├── memory/                     #   Runtime state (не в Git)
└── prompts/                    #   Шаблоны промптов
```

## Как работают скиллы

1. Агент определяет тип задачи.
2. Читает релевантные `SKILL.md` через `view_file`.
3. Применяет стандарты из скиллов при выполнении.
4. После завершения обновляет `knowledge/rules.md`.

Полная таблица маппинга задача→скилл: [[knowledge/rules]].

## Ключевые файлы агента

| Файл | Назначение |
|------|-----------|
| `knowledge/rules.md` | Специфичные правила проекта + отвергнутые подходы |
| `knowledge/glossary.md` | Терминология проекта |
| `knowledge/stack.md` | Все зависимости и технологии |
| `knowledge/mistakes.md` | Лог ошибок с root cause (создаётся по необходимости) |
| `docs/decisions/log.md` | Архитектурные решения (ADR) |
