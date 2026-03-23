"""
Морфологические метрики из 3D voxel-кубов.

Извлекает количественные параметры формы для классификации:
  - volume, sphericity, convexity, surface_roughness, eccentricity
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull


def compute_volume(voxel: np.ndarray, threshold: float = 0.5) -> float:
    """Объём объекта = количество voxel'ей выше порога."""
    return float(np.sum(voxel > threshold))


def compute_surface_area(voxel: np.ndarray, threshold: float = 0.5) -> float:
    """Площадь поверхности через подсчёт граничных voxel'ей.

    Граничный voxel = voxel объекта, у которого хотя бы один сосед — фон.
    """
    binary = (voxel > threshold).astype(np.uint8)
    eroded = ndimage.binary_erosion(binary).astype(np.uint8)
    surface = binary - eroded
    return float(np.sum(surface))


def compute_sphericity(volume: float, surface_area: float) -> float:
    """Sphericity = (π^(1/3) * (6V)^(2/3)) / A.

    Для идеальной сферы sphericity = 1.0.
    """
    if surface_area < 1e-6:
        return 0.0
    numerator = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3))
    return float(numerator / surface_area)


def compute_convexity(voxel: np.ndarray, threshold: float = 0.5) -> float:
    """Convexity = volume / convex_hull_volume.

    Для выпуклого объекта convexity = 1.0.
    Чем больше впадин/отверстий, тем ниже.
    """
    coords = np.argwhere(voxel > threshold)
    if len(coords) < 4:
        return 0.0
    try:
        hull = ConvexHull(coords)
        hull_volume = hull.volume
        actual_volume = float(len(coords))
        if hull_volume < 1e-6:
            return 0.0
        return float(actual_volume / hull_volume)
    except Exception:
        return 0.0


def compute_eccentricity(voxel: np.ndarray, threshold: float = 0.5) -> float:
    """Эксцентриситет через PCA на координатах объекта.

    Отношение наименьшего к наибольшему собственному значению.
    Для идеальной сферы ≈ 1.0, для вытянутого объекта → 0.
    """
    coords = np.argwhere(voxel > threshold).astype(np.float64)
    if len(coords) < 10:
        return 0.0

    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    eigenvalues = np.sort(eigenvalues)

    return float(eigenvalues[0] / eigenvalues[-1])


def compute_surface_roughness(voxel: np.ndarray, threshold: float = 0.5) -> float:
    """Шероховатость поверхности: std расстояний поверхностных voxel'ей от центра масс.

    Для гладкой формы → низкая. Для бугристой → высокая.
    """
    binary = (voxel > threshold).astype(np.uint8)
    eroded = ndimage.binary_erosion(binary).astype(np.uint8)
    surface_coords = np.argwhere((binary - eroded) > 0).astype(np.float64)

    if len(surface_coords) < 10:
        return 0.0

    centroid = surface_coords.mean(axis=0)
    distances = np.linalg.norm(surface_coords - centroid, axis=1)

    return float(np.std(distances))


def extract_all_metrics(
    voxel: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Извлекает все морфологические метрики из 3D voxel-куба.

    Returns:
        dict с ключами: volume, surface_area, sphericity, convexity,
                        eccentricity, surface_roughness
    """
    volume = compute_volume(voxel, threshold)
    surface_area = compute_surface_area(voxel, threshold)
    sphericity = compute_sphericity(volume, surface_area)
    convexity = compute_convexity(voxel, threshold)
    eccentricity = compute_eccentricity(voxel, threshold)
    roughness = compute_surface_roughness(voxel, threshold)

    return {
        "volume": volume,
        "surface_area": surface_area,
        "sphericity": sphericity,
        "convexity": convexity,
        "eccentricity": eccentricity,
        "surface_roughness": roughness,
    }
