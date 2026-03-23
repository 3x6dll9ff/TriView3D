# Архитектура пайплайна

## Общая схема

```
                          DUAL-VIEW 3D CELL SHAPE PREDICTION & CLASSIFICATION

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Данные (SHAPR)                                                            │
│  ─────────────                                                             │
│  obj/ (3D ground truth 64³) ──► извлекаем top/bottom срезы                 │
│                                  │                                          │
│                    ┌──────────────┴──────────────┐                          │
│                    ▼                              ▼                          │
│              Top slice (64×64)           Bottom slice (64×64)               │
│                    │                              │                          │
│                    └──────────┬───────────────────┘                          │
│                               ▼                                             │
│                     ┌─────────────────┐                                     │
│                     │  Dual Input      │                                     │
│                     │  [2, 64, 64]     │                                     │
│                     └────────┬────────┘                                     │
│                              │                                              │
│  ┌───────────────────────────┼───────────────────────────┐                  │
│  │          DUAL-VIEW AUTOENCODER                        │                  │
│  │                           │                            │                  │
│  │  ┌───────────────────────────────────────┐            │                  │
│  │  │  2D Encoder (ConvNet)                  │            │                  │
│  │  │  [2,64,64] → [32,32,32] → [64,16,16]  │            │                  │
│  │  │  → [128,8,8] → [256,4,4] → FC → 256   │            │                  │
│  │  └──────────────────┬────────────────────┘            │                  │
│  │                     │                                  │                  │
│  │              latent vector (256-dim) ──────────► Классификатор           │
│  │                     │                              (MLP)                 │
│  │  ┌──────────────────┴────────────────────┐            │                  │
│  │  │  3D Decoder (ConvTranspose3d)          │            │                  │
│  │  │  256 → [256,4,4,4] → [128,8,8,8]      │            │                  │
│  │  │  → [64,16,16,16] → [32,32,32,32]      │            │                  │
│  │  │  → [1,64,64,64] + Sigmoid              │            │                  │
│  │  └──────────────────┬────────────────────┘            │                  │
│  └───────────────────────┼───────────────────────────────┘                  │
│                          │                                                  │
│                          ▼                                                  │
│                Predicted 3D shape                                           │
│                  [1, 64, 64, 64]                                            │
│                          │                                                  │
│          ┌───────────────┼───────────────┐                                  │
│          ▼               ▼               ▼                                  │
│     Loss (BCE+Dice)  Morphometrics   Визуализация                          │
│     vs Ground Truth   (volume,       (Plotly 3D,                            │
│                       sphericity,    Grad-CAM)                              │
│                       convexity)                                            │
│                          │                                                  │
│                          ▼                                                  │
│                   Random Forest                                             │
│                   (baseline)                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Модели

### 1. Dual-View Autoencoder (основная)
- **Encoder:** 4 слоя Conv2d + BN + ReLU → FC → latent 256-dim
- **Decoder:** FC → reshape → 4 слоя ConvTranspose3d + BN + ReLU → Sigmoid
- **Loss:** 0.5 × BCE + 0.5 × Dice
- **Вход:** [batch, 2, 64, 64] (top + bottom)
- **Выход:** [batch, 1, 64, 64, 64]

### 2. Latent Classifier (MLP)
- 3 слоя FC + ReLU + Dropout
- **Вход:** latent vector 256-dim от обученного autoencoder
- **Выход:** 2 класса (normal / anomaly)

### 3. Random Forest (baseline)
- На морфометрических метриках: volume, sphericity, convexity, eccentricity, surface_roughness
- 200 деревьев, balanced class weight

### 4. SingleView Autoencoder (baseline)
- Тот же autoencoder, но вход [1, 64, 64] (один срез)
- Для доказательства: 2 среза > 1 среза

## Метрики

### Реконструкция
- **Dice score** — overlap predicted vs ground truth
- **IoU** — intersection over union
- **MSE** — mean squared error

### Классификация
- **Accuracy, AUROC, F1, Confusion Matrix**
