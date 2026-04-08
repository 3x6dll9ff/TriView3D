---
name: backend
description: Senior backend engineering standards — Python, FastAPI, API design, security, performance
---

# Backend — Senior Engineer Standards

Все правила ниже обязательны при работе с `src/api.py`, `Dockerfile.backend`, `docker-compose.yml`.

## Philosophy
- Boring is good: используй проверенные паттерны, а не модные.
- Design for failure: каждый внешний вызов может и будет падать.
- Make it observable: если не можешь измерить — не сможешь починить.

## FastAPI Standards (для этого проекта)

### Структура эндпоинтов
```python
@app.get("/api/v1/cells")
def list_cells() -> dict:
    """Docstring обязателен на каждом эндпоинте."""
    ...
```

### Response Shape
Единая структура ответа на всех эндпоинтах:
```python
{
    "data": T | None,      # полезные данные
    "error": str | None,   # ошибка (null если OK)
    "meta": { ... }        # пагинация, версия, timing
}
```
- Исключение: health endpoint может возвращать `{"status": "ok"}`.

### HTTP Status Codes
| Код | Когда |
|-----|-------|
| 200 | Успешный GET/POST |
| 201 | Ресурс создан |
| 400 | Невалидный запрос |
| 404 | Файл/клетка не найдены |
| 422 | Валидация Pydantic провалена |
| 500 | Внутренняя ошибка (модель не загружена, numpy crash) |

### Error Handling
```python
# ✅ Правильно
try:
    top = np.load(path)
except FileNotFoundError:
    raise HTTPException(404, detail="Projection file not found")

# ❌ Неправильно
except Exception as e:
    raise HTTPException(404, detail=str(e))  # утечка внутренних деталей
```
- Никогда не отдавать клиенту `str(e)` — это может содержать путь к файлу, stack trace.
- Логировать полную ошибку внутри, отдавать generic message наружу.

## Python Standards

### Type Hints
На **каждой** функции: аргументы + return type:
```python
def compute_overlap_metrics(pred_vol: np.ndarray, gt_vol: np.ndarray) -> dict[str, float]:
```

### Path Handling
```python
# ✅ Правильно (в новом коде)
from pathlib import Path
data_dir = Path("data/processed")
top = np.load(data_dir / "top_proj" / filename)

# ❌ Устаревший стиль (допускается только в существующем коде)
os.path.join(DATA_DIR, "top_proj", filename)
```
При касании существующего кода — мигрировать на `pathlib`.

### Resource Management
```python
# ✅ Правильно
with open(history_path, "r") as f:
    data = json.load(f)

# ❌ Неправильно
f = open(path)
data = json.load(f)
# f никогда не закрывается
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# ✅ В production-коде (api.py)
logger.info("Model loaded on %s", device)
logger.error("Failed to load projections: %s", filename)

# ✅ В скриптах обучения допускается
print(f"Epoch {epoch} | dice: {dice:.4f}")
```

### Exception Hierarchy
- Никогда голый `except:` — ловить конкретные исключения.
- В API: `HTTPException` с правильным кодом.
- В скриптах: пусть падает с traceback — fail fast.

## Security

### Текущий проект (development)
- CORS: `allow_origins=["*"]` допустим **только** для development.
- Для production: `allow_origins=["http://localhost:5173"]`.

### Общие правила
- Никогда не логировать: пароли, токены, PII.
- Все secrets — через environment variables: `os.environ["VAR"]`.
- Не хардкодить пути к файлам модели — использовать env или конфиг.
- Валидировать все входные данные: `filename` может содержать `../`.
```python
# ✅ Защита от path traversal
from pathlib import Path
safe_path = (Path(DATA_DIR) / "top_proj" / filename).resolve()
if not safe_path.is_relative_to(Path(DATA_DIR).resolve()):
    raise HTTPException(400, "Invalid filename")
```

## Docker

### Dockerfile Best Practices
- Multi-stage builds для production.
- `COPY requirements.txt .` перед `COPY src/` — кэширование слоёв.
- Явные `EXPOSE` порты.
- Healthcheck в docker-compose, не в Dockerfile.
- `.dockerignore` актуален: исключает `data/`, `results/`, `notebooks/`, `docs/`.

### docker-compose.yml
- Сервисы: `backend` (FastAPI) + `frontend` (Vite).
- `restart: unless-stopped` на всех сервисах.
- Frontend зависит от backend через `depends_on` + healthcheck.
- Volumes: монтировать код для dev, COPY для production.

## Performance

### Model Inference
- `model.eval()` + `torch.no_grad()` — **всегда** при инференсе.
- `torch.load(..., map_location=device)` — явно указывать device.
- Для heavy operations (marching cubes): профилировать, кэшировать если возможно.

### API
- Endpoint latency budget: < 500ms для инференса на одной клетке.
- Для тяжёлых операций: `async` + background tasks.
- Не загружать модель заново на каждый запрос — singleton при startup.

## Code Structure

### Файлы src/
```
src/
├── api.py                    # FastAPI endpoints (web layer)
├── autoencoder.py            # Model definitions (domain)
├── classifier.py             # Classification models
├── dataset.py                # PyTorch Dataset (data layer)
├── download_data.py          # Data acquisition script
├── evaluate.py               # Evaluation metrics
├── morphometrics.py          # 3D shape metrics
├── prepare_dataset.py        # Data preprocessing
├── train_classifier.py       # Classifier training
├── train_reconstruction.py   # Autoencoder training
├── visualize.py              # Visualization utilities
└── visualize_examples.py     # Example visualizations
```

### Принцип разделения
- `api.py` — только HTTP layer, вызывает функции из других модулей.
- Бизнес-логика (метрики, трансформации) — отдельные модули.
- Модели — `autoencoder.py`, `classifier.py`.
- Данные — `dataset.py`, `prepare_dataset.py`.

---

## Graceful Degradation

### Что если модель не загружена?
```python
# ❌ Плохо: 500 Internal Server Error с cryptic message
if not model:
    raise HTTPException(500, "Model not loaded")

# ✅ Хорошо: информативный ответ + альтернатива
@app.post("/api/predict/{filename}")
def predict(filename: str):
    if model is None:
        return {
            "data": None,
            "error": "Model not loaded. Upload best_autoencoder.pt to results/",
            "meta": {"model_available": False, "gt_available": check_gt_exists(filename)}
        }
```

### Матрица деградации
| Условие | Поведение | HTTP код |
|---------|-----------|----------|
| Модель загружена, данные есть | Полный ответ (pred + gt + metrics) | 200 |
| Модель загружена, нет GT | Prediction без метрик сравнения | 200 |
| Модель загружена, нет проекций | 404 с информативным сообщением | 404 |
| Модель не загружена | 503 Service Unavailable | 503 |
| OOM при инференсе | 500 + логирование + очистка GPU | 500 |

### GPU Errors в API
```python
@app.post("/api/predict/{filename}")
def predict(filename: str):
    try:
        with torch.no_grad():
            pred = model(input_tensor)
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            logger.error("GPU OOM during inference for %s", filename)
            raise HTTPException(503, detail="GPU memory exhausted. Try again.")
        raise HTTPException(500, detail="Inference failed")
```

## Pydantic Models (для строгой типизации API)

```python
from pydantic import BaseModel, Field

class CellInfo(BaseModel):
    filename: str
    score: str
    type: str

class MeshData(BaseModel):
    vertices: list[float]
    indices: list[int]

class OverlapMetrics(BaseModel):
    dice: float = Field(ge=0, le=1)
    iou: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    volume_diff_pct: float = Field(ge=0)

class PredictionResponse(BaseModel):
    data: MeshData | None
    gt: MeshData | None
    metrics: OverlapMetrics
    error: str | None = None
```
**Зачем**: Pydantic даёт автодокументацию (/docs), валидацию, сериализацию. FastAPI без Pydantic — это Flask с extra steps.

## Request Tracing

```python
import uuid
from fastapi import Request

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    logger.info("→ %s %s [%s]", request.method, request.url.path, request_id)
    
    response = await call_next(request)
    
    logger.info("← %s %s [%s] %d", request.method, request.url.path, request_id, response.status_code)
    response.headers["X-Request-ID"] = request_id
    return response
```
**Зачем**: когда на защите что-то упадёт — по request_id в логах за 10 секунд найдёшь причину.

## Startup / Shutdown Lifecycle

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Loading model from %s", MODEL_PATH)
    global model
    model = load_model()
    logger.info("Model loaded: %d parameters", count_params(model))
    
    yield  # App is running
    
    # SHUTDOWN
    logger.info("Shutting down, cleaning up GPU memory")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)  # Заменяет deprecated @app.on_event
```

## Input Validation (не доверяй ничему)

```python
import re
from pathlib import Path

FILENAME_PATTERN = re.compile(r'^[\w\-\.]+\.npy$')

def validate_filename(filename: str) -> Path:
    """Validate and sanitize filename to prevent path traversal."""
    # 1. Regex whitelist
    if not FILENAME_PATTERN.match(filename):
        raise HTTPException(400, "Invalid filename format")
    
    # 2. Path traversal check
    safe_path = (Path(DATA_DIR) / "top_proj" / filename).resolve()
    if not safe_path.is_relative_to(Path(DATA_DIR).resolve()):
        raise HTTPException(400, "Path traversal detected")
    
    # 3. File existence
    if not safe_path.exists():
        raise HTTPException(404, "File not found")
    
    return safe_path
```

## Latency Budgets

| Endpoint | Target p50 | Target p95 | Budget |
|----------|-----------|-----------|--------|
| `GET /health` | < 5ms | < 10ms | No I/O |
| `GET /api/cells` | < 50ms | < 100ms | Filesystem scan |
| `GET /api/preview/{id}` | < 100ms | < 200ms | 3× np.load + PNG encode |
| `POST /api/predict/{id}` | < 300ms | < 500ms | np.load + inference + marching_cubes |
| `GET /api/metrics` | < 20ms | < 50ms | JSON file read |

### Как измерять
```python
import time

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    if elapsed_ms > 500:
        logger.warning("Slow request: %s %s took %.0fms", request.method, request.url.path, elapsed_ms)
    return response
```

## Dependency Injection (для тестируемости)

```python
# ❌ Глобальная переменная — нетестируемо
model = None

# ✅ Dependency injection через FastAPI
def get_model() -> TriViewAutoencoder:
    if app.state.model is None:
        raise HTTPException(503, "Model not loaded")
    return app.state.model

@app.post("/api/predict/{filename}")
def predict(filename: str, model: TriViewAutoencoder = Depends(get_model)):
    ...

# В тестах:
app.dependency_overrides[get_model] = lambda: mock_model
```

