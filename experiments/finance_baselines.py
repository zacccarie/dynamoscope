"""Phase 6B : finance-specific baselines for Sprint 6 comparison.

Standard regime-switching econometrics + modern sequence models :

  1. HMMBaseline       : Gaussian HMM K=3 (Hamilton 1989 lineage)
  2. MarkovSwitchBaseline : statsmodels MarkovRegression (per-regime AR(1))
  3. TransformerBaseline : 1-layer 4-head Transformer encoder + linear head

Each implements the same eval interface as FlatBaseline / RegimeWorldModel :
  - .forward(x: torch.Tensor) → dict with z_seq, z_pred (for trainer compat)
  - rollout(x_context, K) → predicted next K latent states

For HMM + MarkovSwitch (non-differentiable), we wrap as torch-compatible
modules that delegate to fitted scikit-learn / statsmodels models.
"""
from __future__ import annotations
import warnings
import numpy as np
import torch
import torch.nn as nn


# =========================================================================
# HMM Baseline — Hamilton-style Gaussian HMM K=3
# =========================================================================

class HMMBaseline(nn.Module):
    """Gaussian HMM with K=3 hidden states, fitted per-trajectory at train,
    used for K-step rollout via Viterbi-emission iteration.

    Per-step prediction = Σ_k P(s_{t+1}=k | s_t) · μ_k where μ_k is the
    mean emission of state k. Crude but standard finance regime model.

    Note: not gradient-trainable. Fit via EM (hmmlearn).
    """

    def __init__(self, d_in: int = 5, n_states: int = 3):
        super().__init__()
        # Dummy parameter so .parameters() returns at least 1 tensor (else
        # the existing trainer / count_params() complain).
        self.dummy = nn.Parameter(torch.zeros(1))
        self.d_in = d_in
        self.n_states = n_states
        self.hmm = None  # fitted by .fit() not by gradient training
        self.fitted = False

    def fit(self, dataset):
        """Fit on concatenated features from training dataset."""
        from hmmlearn.hmm import GaussianHMM
        all_feats = np.concatenate([s.traj.astype(np.float32) for s in dataset],
                                     axis=0)
        lengths = [s.traj.shape[0] for s in dataset]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hmm = GaussianHMM(n_components=self.n_states,
                                    covariance_type="diag",
                                    n_iter=50, random_state=0)
            self.hmm.fit(all_feats, lengths=lengths)
        self.fitted = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        """Returns z_seq (= input) + z_pred + z_slow (alias) for eval compat."""
        x_np = x.cpu().numpy().astype(np.float32)
        T = x_np.shape[0]
        if not self.fitted:
            z_pred = torch.zeros_like(x)
            return {"z_seq": x.clone(), "z_pred": z_pred, "x_recon": x.clone(),
                     "z_slow": x.clone(), "r": torch.ones(self.n_states) / self.n_states}
        posteriors = self.hmm.predict_proba(x_np)
        next_state_probs = posteriors @ self.hmm.transmat_
        z_pred_np = next_state_probs @ self.hmm.means_
        z_pred = torch.from_numpy(z_pred_np).float().to(x.device)
        return {"z_seq": x.clone(), "z_pred": z_pred, "x_recon": x.clone(),
                 "z_slow": x.clone(),
                 "r": torch.from_numpy(posteriors).float().to(x.device)}

    def proj_next(self, h: torch.Tensor) -> torch.Tensor:
        """Single-step transition for flat-eval compatibility : h is a
        single state (d_in,). Returns expected next emission via HMM
        transition + means.
        """
        if not self.fitted:
            return h.clone()
        # h shape : (d_in,) — treat as a 'state' for which we don't know
        # the regime, so use uniform posterior, advance once via trans.
        K = self.n_states
        cur = np.ones(K) / K
        next_probs = cur @ self.hmm.transmat_
        mean = next_probs @ self.hmm.means_
        return torch.from_numpy(mean).float().to(h.device)

    @torch.no_grad()
    def rollout(self, x_context: torch.Tensor, K: int) -> dict:
        """K-step rollout : iterate the Markov chain forward + emit means."""
        x_np = x_context.cpu().numpy().astype(np.float32)
        if not self.fitted:
            return {"z_slow_pred": torch.zeros(K, self.d_in)}
        # Start from posterior at last context step
        posteriors = self.hmm.predict_proba(x_np)
        cur_state_probs = posteriors[-1]  # (K,)
        trans = self.hmm.transmat_
        means = self.hmm.means_
        preds = []
        for _ in range(K):
            cur_state_probs = cur_state_probs @ trans
            mean = cur_state_probs @ means  # (d_in,)
            preds.append(torch.from_numpy(mean).float())
        return {"z_slow_pred": torch.stack(preds).to(x_context.device)}


# =========================================================================
# Markov-Switching Baseline — statsmodels MarkovRegression
# =========================================================================

class MarkovSwitchBaseline(nn.Module):
    """K-state Markov regime-switching AR(1) on log-returns (feature 0).

    Per-regime mean + variance. Forecast next return via regime probability
    weighted mean. Multi-step rollout iterates the regime chain forward.

    Single-output (log return) → repeat across all 5 features for shape compat.
    """

    def __init__(self, d_in: int = 5, n_states: int = 3):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.d_in = d_in
        self.n_states = n_states
        self.model = None
        self.result = None
        self.fitted = False

    def fit(self, dataset):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        # Concatenate log_return (feature 0) across all training trajectories
        rets = np.concatenate([s.traj[:, 0].astype(float) for s in dataset])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model = MarkovRegression(
                    rets, k_regimes=self.n_states,
                    switching_variance=True,
                )
                self.result = self.model.fit(disp=False, maxiter=50)
                self.fitted = True
            except Exception as e:
                print(f"  [warn] MarkovSwitch fit failed: {e}")
                self.fitted = False

    def proj_next(self, h: torch.Tensor) -> torch.Tensor:
        """Single-step forecast for flat-eval. Uses fitted regime means."""
        if not self.fitted or self.result is None:
            return h.clone()
        means = self.result.params[:self.n_states]
        cur = np.ones(self.n_states) / self.n_states
        try:
            trans = self.result.regime_transition.squeeze()
            if trans.ndim == 3: trans = trans[:, :, 0]
        except Exception:
            trans = np.eye(self.n_states) * 0.9 + 0.05
        cur = cur @ trans
        ret = cur @ means
        row = h.cpu().numpy().astype(np.float32).copy()
        row[0] = ret
        return torch.from_numpy(row).float().to(h.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        T = x.shape[0]
        if not self.fitted or self.result is None:
            return {"z_seq": x.clone(), "z_pred": torch.zeros_like(x),
                     "x_recon": x.clone(), "z_slow": x.clone(),
                     "r": torch.ones(self.n_states) / self.n_states}
        # Expected return per step = sum_k P(s=k|past) · μ_k
        means = self.result.params[:self.n_states]  # regime means
        # Use smoothed marginal probabilities, then advance via transition.
        smoothed = self.result.smoothed_marginal_probabilities
        if hasattr(smoothed, "values"):
            smoothed = smoothed.values
        # Cut/pad to T (smoothed has shape (N_total, K) from training cat)
        K = self.n_states
        if smoothed.shape[0] < T:
            cur = np.tile(smoothed[-1:], (T, 1))
        else:
            cur = smoothed[-T:]
        try:
            trans = self.result.regime_transition.squeeze()
            if trans.ndim == 3: trans = trans[:, :, 0]
        except Exception:
            trans = np.eye(K) * 0.9 + np.ones((K, K)) * 0.1 / K
        next_probs = cur @ trans
        ret_pred_1d = next_probs @ means  # (T,) — predicted log-return
        # Broadcast to d_in-dim (only feature 0 predicted, rest = input)
        z_pred_np = np.tile(ret_pred_1d[:, None], (1, self.d_in))
        z_pred_np[:, 1:] = x[:, 1:].cpu().numpy()  # passthrough other features
        z_pred = torch.from_numpy(z_pred_np).float().to(x.device)
        return {"z_seq": x.clone(), "z_pred": z_pred, "x_recon": x.clone(),
                 "z_slow": x.clone(),
                 "r": torch.from_numpy(cur.astype(np.float32)).to(x.device)}

    @torch.no_grad()
    def rollout(self, x_context: torch.Tensor, K: int) -> dict:
        """K-step return-only rollout (other features held at last context)."""
        if not self.fitted or self.result is None:
            return {"z_slow_pred": torch.zeros(K, self.d_in)}
        means = self.result.params[:self.n_states]
        try:
            trans = self.result.regime_transition.squeeze()
            if trans.ndim == 3: trans = trans[:, :, 0]
        except Exception:
            trans = np.eye(self.n_states) * 0.9 + 0.05
        smoothed = self.result.smoothed_marginal_probabilities
        if hasattr(smoothed, "values"):
            smoothed = smoothed.values
        cur_probs = smoothed[-1] if smoothed.shape[0] > 0 else np.ones(self.n_states) / self.n_states
        last_ctx = x_context[-1].cpu().numpy()
        preds = []
        for _ in range(K):
            cur_probs = cur_probs @ trans
            ret = cur_probs @ means
            row = np.tile(last_ctx, 1).astype(np.float32)
            row[0] = ret
            preds.append(torch.from_numpy(row).float())
        return {"z_slow_pred": torch.stack(preds).to(x_context.device)}


# =========================================================================
# Transformer Baseline — 1-layer encoder, autoregressive prediction
# =========================================================================

class TransformerBaseline(nn.Module):
    """Small Transformer encoder + linear head for next-step prediction.

    1 layer × 4 heads × d_model=64 ≈ 30K parameters for matched capacity.
    Standard modern sequence baseline.
    """

    def __init__(self, d_in: int = 5, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1):
        super().__init__()
        self.encoder_in = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=64, dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj_next = nn.Linear(d_model, d_model)
        self.decoder = nn.Linear(d_model, d_in)
        self.d_h = d_model

    def forward(self, x: torch.Tensor) -> dict:
        # x : (T, d_in). Add batch dim.
        z = self.encoder_in(x).unsqueeze(0)         # (1, T, d_model)
        # Causal mask : token t cannot attend to t+1, ..., T-1
        T = z.shape[1]
        mask = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool),
                           diagonal=1)
        out = self.transformer(z, mask=mask)       # (1, T, d_model)
        out = out.squeeze(0)                        # (T, d_model)
        z_pred = self.proj_next(out)
        x_recon = self.decoder(out)
        return {"z_seq": out, "z_pred": z_pred, "x_recon": x_recon}
