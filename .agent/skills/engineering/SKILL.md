---
name: engineering
description: General senior engineering standards — Git, ADR, testing, documentation, AI-assisted development
---

# General Senior Engineering Standards

Фундаментальные правила разработки, применимые к любой задаче в проекте.

## Code Quality

### Принципы
- **Boy Scout Rule**: код после тебя чище, чем до тебя.
- **Single Responsibility**: одна функция = одна задача. Макс. ~30 строк, макс. 3 параметра.
- **DRY, но не WET**: дублирование — баг. Но не абстрагируй то, что не повторяется 3+ раз.

### Naming
```python
# ✅ Имя говорит что это
pred_volume = model(tri_input)
is_anomaly = label == 1
train_dice_history = []

# ❌ Абстрактные имена
x = model(inp)
flag = l == 1
arr = []
```

### Comments
```python
# ✅ Комментарий объясняет ПОЧЕМУ
# Smooth=1 предотвращает деление на ноль при пустых масках
dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

# ❌ Комментарий повторяет код
# Вычисляем dice
dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
```
- Если нужен комментарий для объяснения ЧТО — переименуй переменную.
- Dead code удаляется немедленно, не комментируется.

### Функции
```python
# ✅ Чистая функция с ясной сигнатурой
def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Dice coefficient для бинарных 3D масок."""
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ❌ Функция-бог, которая делает всё
def process(data, mode, flag1, flag2, output_path, verbose=False):
    ...  # 200 строк
```

## Git Discipline

### Commits
- Atomic: один коммит = одна логическая единица.
- Format: `<type>(<scope>): <description>`
```
feat(model): add tri-view autoencoder architecture
fix(api): handle missing projection files gracefully
refactor(dataset): migrate os.path to pathlib
docs(architecture): document encoder-decoder pipeline
test(metrics): add unit tests for dice and iou
chore(docker): update base image to python 3.11
```

### Что коммитить
| ✅ Коммитить | ❌ Не коммитить |
|-------------|----------------|
| `src/*.py` | `data/raw/`, `data/processed/` |
| `docs/**/*.md` | `*.pt`, `*.pth`, `*.ckpt` |
| `requirements.txt` | `__pycache__/`, `.venv/` |
| `Dockerfile`, `docker-compose.yml` | `node_modules/`, `dist/` |
| `notebooks/*.ipynb` | `.env`, secrets |
| `.gitignore` | `.DS_Store` |

### Branches
```
main              — стабильный код
feature/xxx       — новая фича
fix/xxx           — исправление бага
chore/xxx         — обслуживание (deps, configs)
```

## Architecture Decision Records (ADR)

### Когда писать
- Смена подхода (intensity layering → autoencoder).
- Смена данных (синтетика → SHAPR).
- Смена архитектуры (dual-view → tri-view).
- Любое решение, которое нетривиально объяснить через 2 месяца.

### Формат (`docs/decisions/log.md`)
```markdown
## N. Название решения
**Дата:** YYYY-MM-DD
**Было:** описание предыдущего подхода
**Стало:** описание нового подхода
**Почему:** обоснование (с данными, если есть)
```

### Текущие ADR проекта
8 решений задокументированы в [[docs/decisions/log]]:
1. Отказ от intensity layering
2. Отказ от синтетических данных
3. Отказ от MONAI DenseNet
4. Метки из имён файлов (не KMeans)
5. Dual-view vs single-view
6. iPSC отдельно от RBC
7. Sum Projection вместо MIP
8. Переход к Tri-View

## Testing

### Приоритет
1. **Unit tests**: чистые функции (`dice_loss`, `compute_overlap_metrics`, `morphometrics`).
2. **Integration tests**: API endpoints через `TestClient`.
3. **Smoke tests**: скрипты обучения запускаются с `--epochs 1`.

### Naming
```python
def test_dice_loss_returns_zero_for_identical_inputs():
    pred = torch.ones(1, 1, 4, 4, 4)
    target = torch.ones(1, 1, 4, 4, 4)
    assert dice_loss(pred, target) < 0.01

def test_api_predict_returns_404_for_missing_file():
    response = client.post("/api/predict/nonexistent.npy")
    assert response.status_code == 404
```

### Coverage
- Цель: **80% на бизнес-логике** (autoencoder, metrics, dataset).
- Не считать покрытие фреймворка (FastAPI routing, PyTorch internals).

## Documentation

### Обязательно
- Каждая публичная функция/класс — docstring.
- `README.md` содержит: описание, быстрый старт, структуру, зависимости.
- Все `.md` файлы имеют YAML frontmatter: `title`, `date`, `tags`, `related`.
- Ссылки между документами — через `[[wikilinks]]`.

### Docstring Format
```python
def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Комбинированный loss: BCE + Dice.

    Args:
        pred: Predicted volume (B, 1, 64, 64, 64), values in [0, 1].
        target: Ground truth volume (B, 1, 64, 64, 64), binary.
        bce_weight: Weight for BCE component.
        dice_weight: Weight for Dice component.

    Returns:
        Scalar loss tensor.
    """
```

## Debugging & Observability

### Logging уровни
| Уровень | Когда |
|---------|-------|
| `DEBUG` | Размерности тензоров, промежуточные значения |
| `INFO` | Старт обучения, загрузка модели, завершение эпохи |
| `WARNING` | Модель не найдена, fallback на CPU |
| `ERROR` | Файл не найден, невалидные данные |

### Что логировать при обучении
- Устройство (GPU/CPU/MPS).
- Число параметров модели.
- Метрики каждой эпохи.
- Время эпохи.
- Сохранение checkpoint/best model.

## AI-Assisted Development Rules

### Перед задачей
1. Прочитать `knowledge/` — понять контекст проекта.
2. Прочитать релевантные `SKILL.md` — применить стандарты.
3. Посмотреть `docs/decisions/log.md` — не повторять отвергнутые подходы.

### После задачи
1. Обновить `knowledge/rules.md` — новые правила или уточнения.
2. Если ошибка — зафиксировать в `knowledge/mistakes.md` с root cause.
3. Если архитектурное решение — добавить ADR в `docs/decisions/log.md`.

### Красные линии
- **Не предполагай** — если задача непонятна, задай вопрос.
- **Не переписывай** большие участки без необходимости.
- **Не ломай** работающий код ради стиля — рефактори при касании.
- **Не добавляй** зависимости без обоснования.

---

## Code Review Checklist

Перед каждым коммитом — пройтись по списку:

### Корректность
- [ ] Код делает то, что задумано (не только happy path)?
- [ ] Edge cases обработаны (пустой массив, NaN, отсутствующий файл)?
- [ ] Нет разрыва контракта: изменения в функции не ломают вызывающий код?

### Безопасность
- [ ] Нет хардкоженных путей, секретов, credentials?
- [ ] Input validation на всех внешних входах (API, файлы, CLI args)?
- [ ] Нет утечки внутренней информации в ответах API?

### Производительность
- [ ] Нет лишних вычислений в hot path (цикл обучения, inference)?
- [ ] Нет N+1 проблем (загрузка файлов в цикле без batch)?
- [ ] `torch.no_grad()` на инференсе?

### Maintainability
- [ ] Другой разработчик (или ты через 2 месяца) поймёт код без объяснений?
- [ ] Нет магических чисел без комментария?
- [ ] Type hints на всех сигнатурах?
- [ ] Docstring на публичных функциях?

### Data Science специфика
- [ ] Seed фиксирован?
- [ ] Train/test split не пересекаются?
- [ ] Метрики считаются на test set, не на train?
- [ ] Tensor shapes аннотированы?

## Refactoring Strategy

### Когда рефакторить
- **Правило трёх**: дублирование кода < 3 раз — OK. ≥ 3 — extract function.
- **При касании**: трогаешь файл → улучши одну мелочь (Boy Scout Rule).
- **Не рефакторь**: за неделю до дедлайна, работающий код без багов, код который не понимаешь.

### Как рефакторить безопасно
```
1. Написать тест на текущее поведение (если нет)
2. Рефакторить маленькими шагами (каждый шаг — рабочий код)
3. Запустить тесты после каждого шага
4. Один коммит = один рефакторинг
```

### Приоритеты рефакторинга в проекте
| Что | Приоритет | Почему |
|-----|-----------|--------|
| `App.tsx` → компоненты | Высокий | Монолит 700+ строк, невозможно поддерживать |
| `api.py` os.path → pathlib | Средний | Inconsistency, но работает |
| `api.py` print → logging | Средний | Нет structured logs |
| `train_reconstruction.py` os.path → pathlib | Низкий | Скрипт, не production |

## Dependency Management

### Добавление зависимости — 5 вопросов
1. **Нужна ли?** Может stdlib / уже установленная библиотека решает?
2. **Поддерживается?** Последний коммит < 6 месяцев, >100 stars?
3. **Лицензия?** MIT/Apache/BSD — OK. GPL — проверить совместимость.
4. **Размер?** Не тянем 50MB ради одной функции.
5. **Конфликты?** Не ломает существующие зависимости.

### requirements.txt
```txt
# ✅ Pin major.minor, не patch
numpy>=1.24.0,<2.0
torch>=2.0.0,<3.0

# ❌ Без пинов — невоспроизводимо
numpy
torch
```

## Pre-commit Checks (рекомендация)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff          # Linting
      - id: ruff-format   # Formatting
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy          # Type checking
```

### Минимальный набор проверок
```bash
# Перед коммитом:
ruff check src/           # Linting
ruff format --check src/  # Formatting
python -c "import src.autoencoder"  # Import check
```

## Trade-off Thinking (что отличает senior)

### Шаблон для решений
```
Проблема: [что нужно решить]
Вариант A: [описание] 
  + [преимущество]
  - [недостаток]
Вариант B: [описание]
  + [преимущество]  
  - [недостаток]
Решение: [A/B] потому что [обоснование в контексте проекта]
```

### Примеры из проекта
```
Проблема: Как подавать проекции в модель?
Вариант A: Binary MIP — проще
  + Одна строка кода
  - Теряет информацию о толщине (все проекции одинаковые)
Вариант B: Sum Projection — сложнее
  + Сохраняет градиент толщины (discocyte = кольцо)
  - Нужна нормализация
Решение: B — потому что информативность проекций критична для качества реконструкции.
→ ADR #7 в docs/decisions/log.md
```

## Incident Response (когда всё сломалось)

### На защите
```
1. Не паниковать. Оценить масштаб (UI? Backend? Данные?)
2. Показать backup: предзаписанное видео / скриншоты результатов
3. Объяснить проблему технически (показывает компетенцию)
4. Если можно починить < 1 мин — чинить. Если нет — backup.
```

### В разработке
```
1. Воспроизвести ошибку (git stash, попробовать на чистом env)
2. Найти root cause (не симптом!)
3. Написать fix + тест, который ловит эту ошибку
4. Зафиксировать в knowledge/mistakes.md
```

