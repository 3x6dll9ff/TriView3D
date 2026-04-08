---
name: frontend
description: Senior frontend engineering standards — React, TypeScript, Vite, Three.js, CSS
---

# Frontend — Senior Engineer Standards

Все правила ниже обязательны при работе с `frontend/`.

## Project Context

Стек фронтенда:
- **React 18** + TypeScript
- **Vite** (dev server + bundler)
- **TailwindCSS** (стилизация)
- **React Three Fiber** (3D визуализация webGL mesh)
- Proxy: Vite → `http://backend:8000` для `/api` запросов

## Architecture

### Текущая структура
```
frontend/
├── src/
│   ├── App.tsx          # Основной компонент (монолит — рефакторить)
│   ├── App.css          # Стили
│   ├── main.tsx         # Entry point
│   ├── index.css        # Global CSS
│   └── assets/          # Статические ресурсы
├── public/              # Favicon, icons
├── index.html           # HTML entry
├── vite.config.ts       # Vite + proxy config
├── tailwind.config.js   # TailwindCSS config
└── tsconfig.json        # TypeScript config
```

### Целевая структура (при рефакторинге)
```
src/
├── components/          # UI-компоненты
│   ├── CellViewer/      # 3D просмотр клетки
│   ├── ProjectionPreview/  # 2D превью проекций
│   ├── MetricsPanel/    # Панель метрик
│   └── CellList/        # Список клеток
├── hooks/               # Кастомные хуки
│   ├── useFetch.ts
│   └── useCellData.ts
├── types/               # TypeScript интерфейсы
│   └── cell.ts
├── utils/               # Утилиты
├── App.tsx
└── main.tsx
```

## React Standards

### Components
```tsx
// ✅ Функциональный компонент с типизированными props
interface CellViewerProps {
  meshData: MeshData | null;
  isLoading: boolean;
  onRotate?: (angle: number) => void;
}

export function CellViewer({ meshData, isLoading, onRotate }: CellViewerProps) {
  // hooks first
  const [rotation, setRotation] = useState(0);

  // handlers
  const handleRotate = useCallback((angle: number) => {
    setRotation(angle);
    onRotate?.(angle);
  }, [onRotate]);

  // early returns
  if (isLoading) return <Spinner />;
  if (!meshData) return <EmptyState />;

  // render
  return (
    <Canvas>
      <CellMesh data={meshData} rotation={rotation} />
    </Canvas>
  );
}
```

### Запрещено
- Class components.
- `any` в TypeScript (если неизбежно — комментарий почему).
- Бизнес-логика в JSX.
- Inline styles (кроме динамических значений).
- `console.log` в production (допускается в dev через env check).

### State Management
Порядок выбора (от простого к сложному):
1. `useState` — локальное состояние компонента
2. `useContext` — shared state между близкими компонентами
3. `zustand` / `jotai` — глобальный state (только если Context недостаточно)

### Custom Hooks
```tsx
// ✅ Логика вынесена в хук
function useCellPrediction(filename: string) {
  const [data, setData] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!filename) return;
    setIsLoading(true);
    fetch(`/api/predict/${filename}`, { method: "POST" })
      .then(res => res.json())
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setIsLoading(false));
  }, [filename]);

  return { data, isLoading, error };
}
```

## TypeScript

### Strict Mode
Всегда `"strict": true` в `tsconfig.json`.

### Types
```tsx
// ✅ Интерфейсы для API responses
interface MeshData {
  vertices: number[];
  indices: number[];
}

interface CellInfo {
  filename: string;
  score: string;
  type: string;
}

interface PredictionResult {
  dice: number;
  metrics: Record<string, number>;
  pred: MeshData | null;
  gt: MeshData | null;
}

// ✅ Props всегда именованные: ComponentNameProps
interface MetricsPanelProps {
  metrics: Record<string, number>;
  isComparing: boolean;
}
```

### Naming
```tsx
// ✅ Descriptive
const isLoading = true;
const handleSubmit = () => {};
const cellList: CellInfo[] = [];

// ❌ Ambiguous
const flag = true;
const click = () => {};
const data: any[] = [];
```

## Styling

### TailwindCSS
- Utility-first: TailwindCSS классы в JSX.
- Кастомные цвета/шрифты — через `tailwind.config.js`.
- Для сложных анимаций — CSS файл, не 10 утилит подряд.

### CSS Custom Properties (если выход за Tailwind)
```css
:root {
  --color-primary: #2563eb;
  --color-surface: #0f172a;
  --color-accent: #06b6d4;
  --radius: 0.5rem;
}
```

### Dark Mode
- Поддерживать через `@media (prefers-color-scheme: dark)` или `data-theme`.
- Все цвета через переменные — никогда хардкод `#fff` / `#000`.

## 3D Visualization (React Three Fiber)

### Специфичные правила для этого проекта
```tsx
// ✅ Mesh из backend-данных
function CellMesh({ vertices, indices }: MeshData) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, [vertices, indices]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color="#06b6d4" />
    </mesh>
  );
}
```
- `useMemo` для геометрии — пересоздание дорогое.
- `computeVertexNormals()` — обязательно для освещения.
- Dispose geometry при unmount (через useEffect cleanup).

## Performance

### Bundle
- Code-split по route: `React.lazy` + `Suspense`.
- Не импортировать всю библиотеку: `import { Canvas } from '@react-three/fiber'`.
- Tree-shaking: использовать named imports.

### Images
- WebP формат, правильные размеры.
- `loading="lazy"` для off-screen изображений.
- Base64 PNG от backend (проекции) — уже оптимально для 64×64.

### 3D Scene
- Ограничивать polygon count — marching cubes даёт адекватный mesh.
- `OrbitControls` с `enableDamping` для плавности.
- Не рендерить невидимые объекты.

## API Communication

### Fetch Pattern
```tsx
// ✅ Единый паттерн для API-вызовов
async function apiCall<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}
```

### Error Handling
- Показывать пользователю: "Модель не загружена" / "Данные не найдены".
- **Не показывать**: stack trace, internal paths, raw error messages.
- Loading states: skeleton / spinner на каждый async operation.

## Motion & Interaction

### CSS Transitions
```css
/* ✅ Для простых state changes */
.panel {
  transition: opacity 200ms ease-out, transform 200ms ease-out;
}
```

### Принципы
- `ease-out` для входов, `ease-in` для выходов.
- `prefers-reduced-motion` — убрать анимации для accessibility.
- 3D rotation: плавный damping через `OrbitControls`.

## Accessibility

- Все интерактивные элементы: `aria-label` если нет видимого текста.
- Keyboard navigation: `tabIndex`, `onKeyDown`.
- Color contrast: WCAG AA минимум.
- Alt text на изображениях проекций.

---

## Error Boundaries

```tsx
// ✅ Обязательно для 3D сцены — WebGL может упасть
class SceneErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("3D Scene crashed:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// Использование:
<SceneErrorBoundary fallback={<div>3D visualization unavailable</div>}>
  <Canvas>
    <CellMesh data={meshData} />
  </Canvas>
</SceneErrorBoundary>
```
**Зачем**: WebGL context loss, GPU OOM в браузере, невалидный mesh — всё это крашит React-дерево без boundary.

## WebGL Context Loss

```tsx
// WebGL контекст может быть потерян (GPU переключение, вкладка в фоне)
function useWebGLRecovery(canvasRef: React.RefObject<HTMLCanvasElement>) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleLost = (e: Event) => {
      e.preventDefault();
      console.warn("WebGL context lost — pausing render");
    };

    const handleRestored = () => {
      console.info("WebGL context restored — resuming render");
      // Re-create geometry, textures
    };

    canvas.addEventListener("webglcontextlost", handleLost);
    canvas.addEventListener("webglcontextrestored", handleRestored);

    return () => {
      canvas.removeEventListener("webglcontextlost", handleLost);
      canvas.removeEventListener("webglcontextrestored", handleRestored);
    };
  }, [canvasRef]);
}
```

## Three.js Memory Management

### Проблема
Каждое переключение клетки создаёт новый `BufferGeometry`. Без cleanup — утечка GPU-памяти.

```tsx
// ✅ Dispose при unmount или prop change
function CellMesh({ vertices, indices }: MeshData) {
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);

  const geometry = useMemo(() => {
    // Dispose предыдущей геометрии
    geometryRef.current?.dispose();

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    
    geometryRef.current = geo;
    return geo;
  }, [vertices, indices]);

  // Cleanup при unmount
  useEffect(() => {
    return () => {
      geometryRef.current?.dispose();
    };
  }, []);

  return <mesh geometry={geometry}><meshStandardMaterial /></mesh>;
}
```

### Мониторинг
```tsx
// В dev mode — отслеживать количество геометрий и текстур
if (import.meta.env.DEV) {
  console.log("Geometries:", renderer.info.memory.geometries);
  console.log("Textures:", renderer.info.memory.textures);
}
```

## Performance Profiling

### React DevTools
- Profiler tab: найти re-renders, которые не должны происходить.
- Highlight updates: увидеть что перерисовывается при каждом state change.

### Оптимизация рендеринга
```tsx
// ✅ Мемоизировать тяжёлые компоненты
const CellList = React.memo(function CellList({ cells }: CellListProps) {
  return cells.map(cell => <CellItem key={cell.filename} cell={cell} />);
});

// ✅ useCallback для handlers, передаваемых в дочерние компоненты
const handleSelect = useCallback((filename: string) => {
  setSelectedCell(filename);
}, []);

// ❌ Не мемоизировать всё подряд — useMemo/useCallback имеют overhead
// Мемоизировать только: тяжёлые вычисления, props для memo-компонентов
```

### Lighthouse Targets
| Метрика | Target | Критично для |
|---------|--------|-------------|
| LCP | < 2.5s | Первая загрузка дашборда |
| FID | < 100ms | Выбор клетки, UI interaction |
| CLS | < 0.1 | Стабильность layout при загрузке 3D |
| TTI | < 3.5s | Полная интерактивность |

## Loading States Hierarchy

```tsx
// ✅ Три уровня loading для UX
type LoadingState = "idle" | "loading" | "error" | "success";

function CellViewer({ filename }: { filename: string }) {
  const { data, isLoading, error } = useCellPrediction(filename);

  // 1. Loading skeleton (не пустое белое окно)
  if (isLoading) return <CellViewerSkeleton />;

  // 2. Информативная ошибка (не "Something went wrong")
  if (error) return (
    <ErrorPanel
      title="Failed to load prediction"
      message={error}
      action={<Button onClick={retry}>Retry</Button>}
    />
  );

  // 3. Empty state (нет данных, но это не ошибка)
  if (!data?.pred) return <EmptyState message="No mesh data available" />;

  // 4. Success
  return <Scene meshData={data.pred} />;
}
```

### Skeleton > Spinner
```tsx
// ❌ Spinner — информации ноль
<div className="flex items-center justify-center h-full">
  <Spinner />
</div>

// ✅ Skeleton — показывает layout, визуально быстрее
<div className="animate-pulse">
  <div className="h-96 bg-gray-800 rounded-lg" />  {/* 3D canvas */}
  <div className="grid grid-cols-3 gap-4 mt-4">
    <div className="h-24 bg-gray-800 rounded" />    {/* Top proj */}
    <div className="h-24 bg-gray-800 rounded" />    {/* Bottom proj */}
    <div className="h-24 bg-gray-800 rounded" />    {/* Side proj */}
  </div>
</div>
```

## Defense Preparation (UX для защиты диплома)

### Что должен делать UI на защите
1. **Произвести впечатление** за первые 5 секунд — 3D модель с вращением.
2. **Показать метрики наглядно** — Dice/IoU с цветовой индикацией (зелёный = хорошо).
3. **Сравнить predicted vs GT** — side-by-side или overlay.
4. **Работать offline** — WiFi в аудитории ненадёжный.

### Offline Resilience
```tsx
// Предзагрузить данные для 5–10 "best" клеток
const DEMO_CELLS = ["0.00_discocyte_123", "1.00_echinocyte_456"];

useEffect(() => {
  // Preload при монтировании
  DEMO_CELLS.forEach(filename => {
    fetch(`/api/predict/${filename}`, { method: "POST" });
    fetch(`/api/preview/${filename}`);
  });
}, []);
```

### Fallback при ошибке
- Модель не загружена → показать pre-rendered screenshots из `results/figures/`.
- Backend упал → показать static JSON с предзагруженными результатами.
- Браузер не поддерживает WebGL → fallback на 2D срезы.

