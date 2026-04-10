import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
import sys

# Добавляем корень проекта для правильных импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoencoder import TriViewAutoencoder
from src.refiner import DetailRefiner
from src.reconstruction_utils import get_view_names, infer_in_channels_from_state_dict, lift_views_to_volume, project_volume_batch
from src.vae import TriViewCVAE, best_of_k_generate

app = FastAPI(title="3D Cell Reconstruction API")

# Настройка CORS для свободного общения с фронтендом на React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/processed"
MODEL_PATH = "results/best_autoencoder.pt"
VAE_MODEL_PATH = "results/best_vae.pt"
REFINER_PATH = "results/best_refiner.pt"

model = None
vae_model = None
refiner_model = None
model_view_names = get_view_names("tri")
vae_view_names = get_view_names("tri")
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def unwrap_state_dict(checkpoint):
    return checkpoint.get("model_state_dict", checkpoint)


def infer_view_names(state_dict):
    in_channels = infer_in_channels_from_state_dict(state_dict)
    return get_view_names("quad" if in_channels == 4 else "tri")


def load_input_views(filename: str, view_names: tuple[str, ...]) -> np.ndarray:
    projections = []
    for view_name in view_names:
        path = os.path.join(DATA_DIR, view_name, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        projections.append(np.load(path))
    return np.stack(projections, axis=0).astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/status")
def get_status():
    return {
        "cnn_loaded": model is not None,
        "vae_loaded": vae_model is not None,
        "refiner_loaded": refiner_model is not None,
    }

@app.on_event("startup")
def load_resources():
    global model, vae_model, refiner_model, model_view_names, vae_view_names
    if os.path.exists(MODEL_PATH):
        print(f"Загрузка CNN модели на {device}...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = unwrap_state_dict(checkpoint)
        model_view_names = infer_view_names(state_dict)
        
        # Автоматический поиск latent_dim
        latent_dim = 256  # fallback
        for key in state_dict.keys():
            if "encoder.fc" in key and "weight" in key:
                latent_dim = int(state_dict[key].shape[0])
                break
                
        model = TriViewAutoencoder(latent_dim=latent_dim, in_channels=len(model_view_names)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print("CNN модель загружена.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: CNN модель не найдена.")

    if model is not None and os.path.exists(REFINER_PATH):
        print(f"Загрузка refiner модели на {device}...")
        refiner_model = DetailRefiner(view_channels=len(model_view_names)).to(device)
        refiner_model.load_state_dict(torch.load(REFINER_PATH, map_location=device))
        refiner_model.eval()
        print("Refiner модель загружена.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Refiner модель не найдена.")

    if os.path.exists(VAE_MODEL_PATH):
        print(f"Загрузка VAE модели на {device}...")
        checkpoint = torch.load(VAE_MODEL_PATH, map_location=device)
        state_dict = unwrap_state_dict(checkpoint)
        vae_view_names = infer_view_names(state_dict)
        latent_dim = int(state_dict["fc_mu.weight"].shape[0])
        vae_model = TriViewCVAE(latent_dim=latent_dim, in_channels=len(vae_view_names)).to(device)
        vae_model.load_state_dict(state_dict)
        vae_model.eval()
        print("VAE модель загружена.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: VAE модель не найдена (обучите через train_vae.py).")

@app.get("/api/cells")
def get_cells():
    """Получает список всех клеток и парсит их метаданные из имени."""
    if not os.path.exists(DATA_DIR):
        return {"cells": []}
        
    files = [f for f in os.listdir(os.path.join(DATA_DIR, "top_proj")) if f.endswith(".npy")]
    result = []
    for f in sorted(files):
        parts = f.replace('.npy', '').split('_')
        score = parts[0]
        c_type = "_".join(parts[1:])
        result.append({
            "filename": f,
            "score": score,
            "type": c_type
        })
    return {"cells": result}

def extract_mesh(volume, level=0.5):
    """Превращает воксели в эффективный 3D-меш для передачи в браузер."""
    try:
        verts, faces, normals, values = marching_cubes(volume, level=level)
        # Для React Three Fiber нам нужен плоский массив координат
        # vertices: [x1,y1,z1, x2,y2,z2...]  и индексы граней
        return {
            "vertices": verts.flatten().tolist(),
            "indices": faces.flatten().tolist()
        }
    except ValueError:
        return None


def compute_overlap_metrics(pred_vol: np.ndarray, gt_vol: np.ndarray) -> dict:
    """Воксельные метрики перекрытия."""
    pred_b = (pred_vol > 0.5).astype(np.float32)
    gt_b = (gt_vol > 0.5).astype(np.float32)

    tp = float(np.sum(pred_b * gt_b))
    fp = float(np.sum(pred_b * (1.0 - gt_b)))
    fn = float(np.sum((1.0 - pred_b) * gt_b))

    dice = (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
    iou = (tp + 1.0) / (tp + fp + fn + 1.0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    pred_voxels = float(np.sum(pred_b))
    gt_voxels = float(np.sum(gt_b))
    volume_diff_pct = (
        abs(pred_voxels - gt_voxels) / (gt_voxels + 1e-8) * 100.0
    )

    return {
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "volume_diff_pct": round(float(volume_diff_pct), 2),
    }


def compute_surface_similarity(pred_mesh: dict | None, gt_mesh: dict | None) -> dict | None:
    """Метрики поверхности по вершинам mesh: ASSD, HD95 и производная similarity."""
    if not pred_mesh or not gt_mesh:
        return None

    pred_v = np.array(pred_mesh["vertices"], dtype=np.float32).reshape(-1, 3)
    gt_v = np.array(gt_mesh["vertices"], dtype=np.float32).reshape(-1, 3)

    if len(pred_v) < 10 or len(gt_v) < 10:
        return None

    max_points = 10000
    if len(pred_v) > max_points:
        idx = np.random.choice(len(pred_v), max_points, replace=False)
        pred_v = pred_v[idx]
    if len(gt_v) > max_points:
        idx = np.random.choice(len(gt_v), max_points, replace=False)
        gt_v = gt_v[idx]

    tree_pred = cKDTree(pred_v)
    tree_gt = cKDTree(gt_v)

    # d_gt_to_pred: для каждой точки GT расстояние до ближайшей точки Pred
    # d_pred_to_gt: для каждой точки Pred расстояние до ближайшей точки GT
    d_gt_to_pred, _ = tree_pred.query(gt_v, k=1)
    d_pred_to_gt, _ = tree_gt.query(pred_v, k=1)

    # ASSD = среднее симметричное поверхностное расстояние (в воксельных единицах)
    assd = float((d_gt_to_pred.mean() + d_pred_to_gt.mean()) / 2.0)

    # HD95 = 95-й перцентиль симметричного Hausdorff (устойчивее к выбросам)
    hd95 = float(
        max(
            np.percentile(d_gt_to_pred, 95),
            np.percentile(d_pred_to_gt, 95),
        )
    )

    # Псевдо-нормированная похожесть: 1/(1+ASSD), ближе к 1 при меньшей ошибке.
    surface_similarity = float(1.0 / (1.0 + assd))

    return {
        "surface_assd": round(assd, 4),
        "surface_hd95": round(hd95, 4),
        "surface_similarity": round(float(surface_similarity), 4),
    }

@app.post("/api/predict/{filename}")
def predict(filename: str):
    """Принимает имя файла, прогоняет через PyTorch автоэнкодер и отдает обратно две 3D-модели."""
    if not model:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

    try:
        multi_view_input = load_input_views(filename, model_view_names)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файлы проекций не найдены.")

    tri_tensor = torch.tensor(multi_view_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        coarse_logits = model(tri_tensor)
        coarse_tensor = torch.sigmoid(coarse_logits)
        coarse_vol = coarse_tensor[0, 0].cpu().numpy()
        
        pred_tensor = coarse_tensor
        if refiner_model is not None:
            lifted_views = lift_views_to_volume(tri_tensor, model_view_names)
            # Передаем сырые логиты (coarse_logits), рефайнер сам сделает sigmoid где нужно
            refined_logits = refiner_model(coarse_logits, lifted_views)
            pred_tensor = torch.sigmoid(refined_logits)
            
        pred_vol = pred_tensor[0, 0].cpu().numpy()
        reprojection_l1 = float(torch.abs(project_volume_batch(pred_tensor, model_view_names) - tri_tensor).mean().item())

    pred_mesh = extract_mesh(pred_vol)
    coarse_mesh = extract_mesh(coarse_vol) if refiner_model is not None else None

    # Чтение Ground Truth для сравнения
    gt_mesh = None
    gt_path = os.path.join(DATA_DIR, "obj", filename)
    if os.path.exists(gt_path):
        gt_vol = np.load(gt_path)
        gt_mesh = extract_mesh(gt_vol)
        
    # Воксельные и поверхностные метрики
    overlap_metrics = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "volume_diff_pct": 0.0,
    }
    surface_metrics = None
    if os.path.exists(gt_path):
        overlap_metrics = compute_overlap_metrics(pred_vol, gt_vol)
        surface_metrics = compute_surface_similarity(pred_mesh, gt_mesh)
        
    return {
        "dice": overlap_metrics["dice"],  # backward compatibility для старого UI
        "metrics": {
            **overlap_metrics,
            **(surface_metrics or {}),
            "reprojection_l1": round(reprojection_l1, 4),
        },
        "pred": pred_mesh,
        "coarse": coarse_mesh,
        "gt": gt_mesh
    }

import json

from PIL import Image
import io
import base64

def numpy_to_b64_png(arr):
    # Убираем NaN
    arr = np.nan_to_num(arr)
    
    # Автоконтраст: растягиваем гистограмму на полный диапазон [0, 255]
    min_val = arr.min()
    max_val = arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        arr = np.zeros_like(arr)
    
    # Конвертируем в uint8
    arr_uint8 = arr.astype(np.uint8)
    
    # Для SHAPR это скорее всего одноканальные 64x64. Убедимся, что размерность [64,64]
    if arr_uint8.ndim == 3 and arr_uint8.shape[0] == 1:
        arr_uint8 = arr_uint8[0]
        
    img = Image.fromarray(arr_uint8, mode='L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

@app.get("/api/preview/{filename}")
def preview_projections(filename: str):
    """Возвращает доступные проекции в виде base64 png для предпросмотра на UI."""
    try:
        response = {}
        for short_name, view_name in [
            ("top", "top_proj"),
            ("bottom", "bottom_proj"),
            ("side", "side_proj"),
            ("front", "front_proj"),
        ]:
            path = os.path.join(DATA_DIR, view_name, filename)
            if os.path.exists(path):
                response[short_name] = numpy_to_b64_png(np.load(path))
        if not response:
            raise FileNotFoundError(filename)
        return response
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Не удалось загрузить проекции: {str(e)}")

@app.get("/api/metrics")
def get_metrics():
    """Возвращает историю лоссов и метрик из JSON для вкладки Metrics."""
    history_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics", "reconstruction_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "History file not found."}


@app.post("/api/predict-vae/{filename}")
def predict_vae(filename: str):
    """Инференс VAE модели — generative подход к реконструкции."""
    if not vae_model:
        raise HTTPException(status_code=500, detail="VAE модель не загружена. Обучите через train_vae.py.")

    try:
        multi_view_input = load_input_views(filename, vae_view_names)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файлы проекций не найдены.")

    tri_tensor = torch.tensor(multi_view_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_tensor, best_score = best_of_k_generate(vae_model, tri_tensor, vae_view_names, num_samples=8)
        pred_vol = pred_tensor[0, 0].cpu().numpy()

    pred_mesh = extract_mesh(pred_vol)

    gt_mesh = None
    gt_path = os.path.join(DATA_DIR, "obj", filename)
    overlap_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "volume_diff_pct": 0.0}
    surface_metrics = None

    if os.path.exists(gt_path):
        gt_vol = np.load(gt_path)
        gt_mesh = extract_mesh(gt_vol)
        overlap_metrics = compute_overlap_metrics(pred_vol, gt_vol)
        surface_metrics = compute_surface_similarity(pred_mesh, gt_mesh)

    return {
        "dice": overlap_metrics["dice"],
        "metrics": {**overlap_metrics, **(surface_metrics or {}), "reprojection_l1": round(float(best_score.item()), 4)},
        "pred": pred_mesh,
        "gt": gt_mesh,
    }


@app.get("/api/metrics-vae")
def get_vae_metrics():
    """История обучения VAE."""
    history_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "metrics", "vae_history.json",
    )
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "VAE history file not found."}

# Точка входа для запуска через `python src/api.py`
if __name__ == "__main__":
    import uvicorn
    # Запускаем локальный веб-сервер на 8000 порту
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
