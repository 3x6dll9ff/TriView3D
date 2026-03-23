"""
Классификатор морфологии: normal vs anomaly.

Два подхода:
  1. MLP на latent vector от autoencoder (end-to-end)
  2. Random Forest на ручных морфометрических метриках (baseline)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


class LatentClassifier(nn.Module):
    """MLP классификатор на latent vector от autoencoder.

    Input:  [batch, latent_dim]
    Output: [batch, 2] — logits для normal / anomaly
    """

    def __init__(self, latent_dim: int = 256, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MorphometryRFClassifier:
    """Random Forest на ручных морфометрических метриках.

    Baseline для сравнения с neural network подходом.
    """

    FEATURE_COLS = [
        "volume",
        "sphericity",
        "convexity",
        "eccentricity",
        "surface_roughness",
    ]

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )

    def fit(self, df, label_col: str = "label") -> "MorphometryRFClassifier":
        X = df[self.FEATURE_COLS].values
        y = df[label_col].values
        self.model.fit(X, y)
        return self

    def predict(self, df) -> np.ndarray:
        X = df[self.FEATURE_COLS].values
        return self.model.predict(X)

    def predict_proba(self, df) -> np.ndarray:
        X = df[self.FEATURE_COLS].values
        return self.model.predict_proba(X)

    def evaluate(self, df, label_col: str = "label") -> dict[str, float]:
        X = df[self.FEATURE_COLS].values
        y_true = df[label_col].values
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auroc": roc_auc_score(y_true, y_proba)
            if len(np.unique(y_true)) > 1
            else 0.0,
            "report": classification_report(y_true, y_pred, output_dict=True),
        }

    def feature_importance(self) -> dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(self.FEATURE_COLS, importances))
