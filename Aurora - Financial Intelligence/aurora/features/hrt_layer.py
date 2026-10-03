"""
Layer 2 — Hamiltonian Regime Tuning (HRT) Feature Layer.

Fuses classical price/macro features with the HMM filtered posterior via a
learned quantum-inspired transformation. The output is a regime-conditioned
feature tensor where every dimension captures non-linear interactions between
market state and price signals.

Mathematical mechanism
----------------------
Given classical features x ∈ R^d and regime posterior γ ∈ R^K:

    combined = Linear([x; γ])  →  [θ_x, θ_γ] ∈ R^{2·n_qubits}
    HRT(x, γ) = cos(θ_x · π) ⊙ sin(θ_γ · π)

This is equivalent to a single-layer Random Fourier Feature (RFF) kernel
approximation conditioned on regime state, capturing multiplicative
feature×regime interactions.

Usage::

    layer = HRTLayer(n_features=50, n_regimes=4, n_qubits=32)
    hrt_features = layer.forward(feature_tensor, regime_posterior_tensor)

    # Convenience wrapper for pandas DataFrames:
    df_hrt = apply_hrt(features_df, regime_df, n_qubits=32)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class HRTLayer(nn.Module):
    """
    Hamiltonian Regime Tuning layer.

    Parameters
    ----------
    n_features : number of classical input features (d)
    n_regimes  : size of the regime posterior vector (K)
    n_qubits   : output dimension (also the "qubit" dimension)
    dropout    : regularisation dropout applied to the combined context
    """

    def __init__(
        self,
        n_features: int,
        n_regimes: int,
        n_qubits: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_regimes = n_regimes
        self.n_qubits = n_qubits

        # Projects [classical_features; regime_posterior] → [θ_x | θ_γ]
        self.projection = nn.Linear(n_features + n_regimes, 2 * n_qubits, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(n_features + n_regimes)

        # Initialise with small weights so the transformation starts near identity
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        x_classical: torch.Tensor,      # (batch, n_features)
        gamma_regime: torch.Tensor,     # (batch, n_regimes)
    ) -> torch.Tensor:                  # (batch, n_qubits)
        combined = torch.cat([x_classical, gamma_regime], dim=-1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        mapped = self.projection(combined)                          # (batch, 2*n_qubits)
        theta_x = mapped[:, :self.n_qubits] * torch.tensor(np.pi) # encoding angles
        theta_g = mapped[:, self.n_qubits:] * torch.tensor(np.pi) # variational angles

        # Quantum-inspired feature: cos(encoding) × sin(variational)
        quantum_features = torch.cos(theta_x) * torch.sin(theta_g)
        return quantum_features                                     # (batch, n_qubits)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        n_features: int,
        n_regimes: int,
        n_qubits: int = 32,
    ) -> "HRTLayer":
        layer = cls(n_features, n_regimes, n_qubits)
        layer.load_state_dict(torch.load(path, map_location="cpu"))
        layer.eval()
        return layer


def apply_hrt(
    features_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_qubits: int = 32,
    hrt_layer: HRTLayer | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Convenience wrapper: applies HRTLayer to pandas DataFrames.

    Parameters
    ----------
    features_df : classical features (rows=dates, cols=features)
    regime_df   : output of OnlineHMMFilter.to_dataframe(); must contain
                  prob_* columns for the posterior vector
    n_qubits    : HRT output dimension
    hrt_layer   : pre-trained HRTLayer; if None, creates a randomly
                  initialised one (useful for building the joint model)
    device      : "cpu" or "cuda"

    Returns
    -------
    DataFrame of HRT features, same DatetimeIndex as the intersection of
    features_df and regime_df.
    """
    # Align on common index
    common_idx = features_df.index.intersection(regime_df.index)
    if len(common_idx) == 0:
        logger.warning("[HRT] Empty intersection between features and regime DataFrames")
        return pd.DataFrame(index=features_df.index)

    feat = features_df.loc[common_idx].ffill().bfill().fillna(0.0)
    prob_cols = [c for c in regime_df.columns if c.startswith("prob_")]
    if not prob_cols:
        logger.warning("[HRT] No prob_* columns found in regime_df — using uniform posterior")
        n_regimes = 4
        gamma = np.ones((len(common_idx), n_regimes)) / n_regimes
    else:
        gamma = regime_df.loc[common_idx, prob_cols].fillna(1.0 / len(prob_cols)).values

    n_features = feat.shape[1]
    n_regimes = gamma.shape[1]

    if hrt_layer is None:
        hrt_layer = HRTLayer(n_features, n_regimes, n_qubits)
        hrt_layer.eval()

    x_t = torch.tensor(feat.values, dtype=torch.float32).to(device)
    g_t = torch.tensor(gamma, dtype=torch.float32).to(device)

    with torch.no_grad():
        hrt_out = hrt_layer(x_t, g_t).cpu().numpy()

    col_names = [f"hrt_{i}" for i in range(n_qubits)]
    hrt_df = pd.DataFrame(hrt_out, index=common_idx, columns=col_names)

    # Concatenate HRT features with original features for a richer representation
    combined = pd.concat([feat, hrt_df], axis=1)
    logger.info(
        f"[HRT] Applied: {n_features} classical + {n_qubits} HRT = {combined.shape[1]} total features"
    )
    return combined
