"""Finance / regime-switching market data for RegimeWorldModel Sprint 5.

Hypothesis (test of domain fit) : the regime-aware architecture should
perform much better on canonical regime-switching domains than on real
video, because :
  - market regimes (bull/bear/sideways) ARE a natural K=3 taxonomy
  - GARCH-style volatility clustering = regime structure baked into data
  - per-timestep router IS what regime-switching econometrics does
  - low-D input (5-30 features) matches our architectural sweet spot

This module provides :

1. gen_regime_switching_market(): synthetic 3-regime market series
   with explicit regime labels, GARCH-like vol clustering, hidden
   Markov regime transitions.

2. MarketSample dataclass : trajectory-of-features + per-step regime
   labels, .traj/.regime compatible with existing trainer interface.

3. make_market_dataset(): N trajectories with controlled regime
   distributions for train/eval.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# Regime parameters : (drift_mu, vol_sigma, persistence)
# Persistence = prob of staying in regime at next step (Markov)
REGIME_PARAMS = {
    "bull":     {"drift": +0.0008,  "vol_base": 0.010, "ar_coef": +0.1},
    "bear":     {"drift": -0.0010,  "vol_base": 0.022, "ar_coef": -0.05},
    "sideways": {"drift":  0.0,     "vol_base": 0.008, "ar_coef": -0.3},  # mean-reverting
}
REGIME_TO_IDX = {"bull": 0, "bear": 1, "sideways": 2}
REGIMES = list(REGIME_PARAMS.keys())

# Hidden Markov transition matrix : rows = from, cols = to.
# Diagonals high → regimes persist (typical 50-200 step periods).
TRANSITION_MATRIX = np.array([
    [0.985, 0.005, 0.010],   # bull → mostly stays, some bear/sideways
    [0.010, 0.980, 0.010],   # bear → mostly stays
    [0.015, 0.010, 0.975],   # sideways → mostly stays, slight bull bias
])


@dataclass
class MarketSample:
    """Market trajectory with multi-feature input + per-step regime label.

    Compatible with trainer (has .traj and .regime as plain fields so the
    experiment script can remap regime names via direct assignment).
    """
    traj: np.ndarray           # (T, F) features per step
    regime_labels: np.ndarray  # (T,) per-step regime int in {0,1,2}
    price: np.ndarray          # (T,) raw close price (for analysis/Sharpe)
    return_series: np.ndarray  # (T,) log returns
    params: dict
    regime: str = "sideways"   # dominant regime, set in __post_init__

    def __post_init__(self):
        if len(self.regime_labels) > 0:
            counts = np.bincount(self.regime_labels, minlength=3)
            self.regime = REGIMES[int(counts.argmax())]


def gen_regime_switching_market(T: int = 1024, seed: int = 0,
                                  initial_price: float = 100.0,
                                  initial_regime: str = "sideways",
                                  ) -> MarketSample:
    """Generate one synthetic market trajectory.

    State each step :
      - hidden regime r_t ∈ {bull, bear, sideways} via Markov chain
      - log-return     u_t ~ drift(r_t) + sigma(r_t) * epsilon_t
                       + ar_coef(r_t) * u_{t-1}
      - vol            sigma_t = vol_base(r_t) * (1 + alpha * |u_{t-1}|)
                       (GARCH-like persistence)
      - price          P_t = P_{t-1} * exp(u_t)
      - volume         V_t = base * exp(0.5 * |u_t| / sigma_t)  (vol up on big moves)

    Output features per step (5-D) :
      [log_return, log_volume, vol_realized_5, rsi_14, sma_diff_20]
    """
    rng = np.random.default_rng(seed)
    P = np.zeros(T); P[0] = initial_price
    returns = np.zeros(T)
    volumes = np.zeros(T)
    regime_labels = np.zeros(T, dtype=np.int64)
    cur_regime_idx = REGIME_TO_IDX[initial_regime]
    prev_u = 0.0
    base_volume = 1000.0
    alpha_garch = 2.0

    for t in range(T):
        # Markov regime transition
        if t > 0:
            probs = TRANSITION_MATRIX[cur_regime_idx]
            cur_regime_idx = int(rng.choice(3, p=probs))
        regime_labels[t] = cur_regime_idx
        params = REGIME_PARAMS[REGIMES[cur_regime_idx]]
        # Vol with GARCH-like dependence on |prev_u|
        sigma = params["vol_base"] * (1 + alpha_garch * abs(prev_u))
        # Log return with AR(1)
        eps = rng.standard_normal()
        u = params["drift"] + params["ar_coef"] * prev_u + sigma * eps
        returns[t] = u
        if t > 0:
            P[t] = P[t - 1] * np.exp(u)
        # Volume up on absolute moves
        V = base_volume * np.exp(0.5 * abs(u) / max(sigma, 1e-6))
        volumes[t] = V + rng.normal(0, base_volume * 0.05)
        prev_u = u

    # Compute features
    log_vol = np.log(np.maximum(volumes, 1.0))
    # Realized vol over 5 steps
    rv5 = np.zeros(T)
    for t in range(T):
        win = returns[max(0, t - 4):t + 1]
        rv5[t] = win.std() if len(win) > 1 else 0.0
    # RSI 14 (simple)
    rsi = np.zeros(T) + 50
    period = 14
    for t in range(period, T):
        rets_win = returns[t - period:t]
        gains = rets_win[rets_win > 0].sum()
        losses = -rets_win[rets_win < 0].sum()
        if losses < 1e-9:
            rsi[t] = 100.0
        else:
            rs = gains / losses
            rsi[t] = 100 - 100 / (1 + rs)
    # SMA(20) difference from price (slope proxy)
    sma_diff = np.zeros(T)
    for t in range(20, T):
        sma = P[t - 20:t].mean()
        sma_diff[t] = (P[t] - sma) / sma
    features = np.stack([returns, log_vol, rv5, rsi / 100.0, sma_diff], axis=1).astype(np.float32)

    return MarketSample(
        traj=features, regime_labels=regime_labels,
        price=P.astype(np.float32),
        return_series=returns.astype(np.float32),
        params={"seed": seed, "T": T, "initial_regime": initial_regime},
    )


def make_market_dataset(n_samples: int = 32, T: int = 512, base_seed: int = 0
                         ) -> list[MarketSample]:
    """N market trajectories with varied initial regime."""
    rng = np.random.default_rng(base_seed)
    out = []
    for i in range(n_samples):
        init = REGIMES[int(rng.integers(0, 3))]
        out.append(gen_regime_switching_market(
            T=T, seed=base_seed + i * 17, initial_regime=init,
        ))
    return out
