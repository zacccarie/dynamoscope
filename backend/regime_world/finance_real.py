"""Real financial data loader for Sprint 6 — replaces synthetic GARCH
trajectories of finance.py by actual market data (BTC-USD, ETH-USD,
SPY, ...).

Same 5-feature output as finance.MarketSample for drop-in compatibility :
  [log_return, log_volume, realized_vol_5, rsi_14, sma_20_dev]

Per-step regime labels are computed via a simple heuristic (trend +
volatility) since real data has no ground-truth regime. This is honest
labelling — we use it only for trainer compat with K=3 supervised CE,
not for evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Reuse the MarketSample shape from synthetic finance module.
from .finance import MarketSample, REGIMES, REGIME_TO_IDX

CACHE_DIR = Path("cache/finance_real")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_yfinance(ticker: str = "BTC-USD", period: str = "5y",
                    interval: str = "1d") -> np.ndarray:
    """Fetch OHLCV via yfinance. Cached on disk.

    Returns (N, 5) array : [Open, High, Low, Close, Volume].
    """
    cache_path = CACHE_DIR / f"{ticker.replace('-', '_')}_{period}_{interval}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    # New yfinance has MultiIndex columns; flatten by taking level 0.
    cols = ["Open", "High", "Low", "Close", "Volume"]
    arr = np.column_stack([df[c].values.squeeze() for c in cols]).astype(np.float32)
    arr = arr[~np.isnan(arr).any(axis=1)]
    np.save(cache_path, arr)
    return arr


def compute_features(ohlcv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OHLCV (N, 5) → (features (N, 5), returns (N,), price (N,)).

    Features per step :
      0. log_return        = log(Close_t / Close_{t-1})
      1. log_volume        = log(Volume_t + 1)
      2. realized_vol_5    = std(returns[t-4:t+1])
      3. rsi_14            = RSI over 14 steps
      4. sma_20_deviation  = (Close - SMA20) / SMA20
    """
    close = ohlcv[:, 3]
    volume = ohlcv[:, 4]
    N = len(close)
    returns = np.zeros(N, dtype=np.float32)
    returns[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-9))
    log_volume = np.log(np.maximum(volume, 1.0)).astype(np.float32)
    # Standardize log_volume per-trajectory for numerical scale
    log_volume = (log_volume - log_volume.mean()) / max(log_volume.std(), 1e-6)
    rv5 = np.zeros(N, dtype=np.float32)
    for t in range(N):
        win = returns[max(0, t - 4):t + 1]
        rv5[t] = win.std() if len(win) > 1 else 0.0
    rsi = np.full(N, 50.0, dtype=np.float32)
    period = 14
    for t in range(period, N):
        rets_win = returns[t - period:t]
        gains = rets_win[rets_win > 0].sum()
        losses = -rets_win[rets_win < 0].sum()
        if losses < 1e-9:
            rsi[t] = 100.0
        else:
            rs = gains / losses
            rsi[t] = 100 - 100 / (1 + rs)
    rsi = rsi / 100.0
    sma_dev = np.zeros(N, dtype=np.float32)
    for t in range(20, N):
        sma = close[t - 20:t].mean()
        sma_dev[t] = (close[t] - sma) / max(sma, 1e-6)
    features = np.stack([returns, log_volume, rv5, rsi, sma_dev], axis=1)
    return features, returns, close.astype(np.float32)


def heuristic_regime_labels(returns: np.ndarray, window: int = 20,
                              vol_threshold_pct: float = 75.0
                              ) -> np.ndarray:
    """Heuristic per-step regime label (bull / sideways / bear).

    bull     : 20-step SMA slope > 0 AND realized vol below 75-th pct
    bear     : 20-step SMA slope < 0 AND realized vol above 75-th pct
    sideways : everything else

    Used as supervisory signal for trainer (K=3 CE). NOT used in eval.
    """
    N = len(returns)
    sma_slope = np.zeros(N)
    for t in range(window, N):
        win = returns[t - window:t]
        # cumulative log-return over window = trend proxy
        sma_slope[t] = win.sum()
    rv = np.zeros(N)
    for t in range(N):
        win = returns[max(0, t - window + 1):t + 1]
        rv[t] = win.std() if len(win) > 1 else 0.0
    vol_thresh = np.percentile(rv[window:], vol_threshold_pct)
    labels = np.full(N, REGIME_TO_IDX["sideways"], dtype=np.int64)
    bull_mask = (sma_slope > 0) & (rv < vol_thresh)
    bear_mask = (sma_slope < 0) & (rv >= vol_thresh)
    labels[bull_mask] = REGIME_TO_IDX["bull"]
    labels[bear_mask] = REGIME_TO_IDX["bear"]
    return labels


def make_real_market_dataset(
    tickers: list[str] = ("BTC-USD", "ETH-USD"),
    period: str = "5y", interval: str = "1d",
    window_T: int = 256, stride: int = 64,
) -> list[MarketSample]:
    """Build dataset : for each ticker, fetch + featurize + slice into
    overlapping windows of length window_T (stride = stride).

    Returns list[MarketSample] compatible with Sprint 5 experiment script.
    """
    out = []
    for ticker in tickers:
        try:
            ohlcv = fetch_yfinance(ticker, period, interval)
        except Exception as e:
            print(f"  [skip] {ticker}: {e}")
            continue
        features, returns, price = compute_features(ohlcv)
        labels = heuristic_regime_labels(returns)
        N = features.shape[0]
        if N < window_T + 4:
            print(f"  [skip] {ticker}: too short ({N} < {window_T})")
            continue
        for start in range(0, N - window_T + 1, stride):
            end = start + window_T
            s = MarketSample(
                traj=features[start:end].copy(),
                regime_labels=labels[start:end].copy(),
                price=price[start:end].copy(),
                return_series=returns[start:end].copy(),
                params={"ticker": ticker, "window_start": int(start),
                         "window_end": int(end)},
            )
            out.append(s)
        print(f"  [{ticker:<8}] {N} candles → {(N - window_T) // stride + 1} windows")
    return out
