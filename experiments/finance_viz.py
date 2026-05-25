"""Phase 6B #4 : router trajectory visualization on BTC/ETH price charts.

Plot price + router output r_t (3-channel, stacked area) + heuristic regime
labels for comparison. Outputs PNG to figures/ directory.

Industrial lab palette (sulfur lime #d4d400, warning red #ff3838, black bg).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

FIG_DIR = Path("figures/finance_router")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Industrial lab palette
COLORS = {
    "smooth": "#5bc97a",       # green for bull
    "periodic": "#d4d400",     # sulfur lime for sideways
    "chaotic": "#ff3838",      # warning red for bear
    "price": "#e0e0e0",        # light gray for price
    "bg": "#0a0a0a",           # near-black background
    "grid": "#2a2a2a",
}


def plot_router_overlay(model, sample, save_name: str = None,
                         title: str = None):
    """Run model.forward(sample.traj) and plot router output over time
    against the price curve and the heuristic regime labels.

    sample : MarketSample with .traj, .price, .regime_labels
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = torch.from_numpy(sample.traj.astype(np.float32))
    x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
    model.eval()
    with torch.no_grad():
        out = model(x)
    r = out["r"]
    # Handle per_timestep router shape (T_s, K) or scalar (K,)
    if r.dim() == 1:
        # Single regime distribution — broadcast over slow time steps.
        T_s = out["z_slow"].shape[0]
        r = r.unsqueeze(0).expand(T_s, -1)
    r_np = r.cpu().numpy()  # (T_s, 3)
    T_s = r_np.shape[0]
    T_raw = sample.traj.shape[0]
    # Upsample r to raw time scale (slow_stride=4)
    t_slow = np.linspace(0, T_raw - 1, T_s)
    t_raw = np.arange(T_raw)
    r_upsampled = np.zeros((T_raw, 3))
    for k in range(3):
        r_upsampled[:, k] = np.interp(t_raw, t_slow, r_np[:, k])

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1.2, 0.8]},
                              facecolor=COLORS["bg"])
    # Panel 1 : price
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    ax.plot(t_raw, sample.price, color=COLORS["price"], linewidth=1.5)
    ax.set_ylabel("Price ($)", color=COLORS["price"], fontsize=10)
    ax.tick_params(colors=COLORS["price"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], alpha=0.4, linewidth=0.5)
    if title:
        ax.set_title(title, color="white", fontsize=12)

    # Panel 2 : router output stacked (RWM regime confidence)
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])
    ax.fill_between(t_raw, 0, r_upsampled[:, 0],
                     color=COLORS["smooth"], alpha=0.8, label="bull (smooth)")
    ax.fill_between(t_raw, r_upsampled[:, 0],
                     r_upsampled[:, 0] + r_upsampled[:, 1],
                     color=COLORS["periodic"], alpha=0.8,
                     label="sideways (periodic)")
    ax.fill_between(t_raw, r_upsampled[:, 0] + r_upsampled[:, 1],
                     r_upsampled[:, 0] + r_upsampled[:, 1] + r_upsampled[:, 2],
                     color=COLORS["chaotic"], alpha=0.8,
                     label="bear (chaotic)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("RWM router r_t", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
              labelcolor="white", fontsize=8, ncol=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])

    # Panel 3 : heuristic labels (ground-truth-ish)
    ax = axes[2]
    ax.set_facecolor(COLORS["bg"])
    label_colors = np.array([
        [COLORS["smooth"], COLORS["chaotic"], COLORS["periodic"]][int(lbl)]
        for lbl in sample.regime_labels
    ])
    # Draw colored bands
    for t in range(T_raw):
        ax.axvspan(t - 0.5, t + 0.5, color=label_colors[t], alpha=0.9, linewidth=0)
    ax.set_yticks([])
    ax.set_xlabel("Time (days)", color="white", fontsize=10)
    ax.set_ylabel("Heuristic", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])

    plt.tight_layout()
    if save_name is None:
        save_name = f"router_{sample.params.get('ticker', 'unknown')}_{sample.params.get('window_start', 0)}.png"
    out_path = FIG_DIR / save_name
    plt.savefig(out_path, dpi=120, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    return out_path


def plot_regime_occupancy(samples, model, save_name: str = "regime_occupancy.png"):
    """Histogram of router-detected regime fractions across many windows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    occupancies = []
    for s in samples:
        x = torch.from_numpy(s.traj.astype(np.float32))
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        with torch.no_grad():
            out = model(x)
        r = out["r"]
        if r.dim() == 1:
            r = r.unsqueeze(0)
        occupancies.append(r.mean(dim=0).cpu().numpy())  # (3,)
    occ = np.stack(occupancies)  # (N, 3)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    labels = ["bull", "sideways", "bear"]
    bar_colors = [COLORS["smooth"], COLORS["periodic"], COLORS["chaotic"]]
    means = occ.mean(axis=0); stds = occ.std(axis=0)
    x_pos = np.arange(3)
    ax.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.85,
            error_kw={"ecolor": "white", "capsize": 5})
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("Mean router occupancy (BTC+ETH)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.4)
    ax.set_title(f"Router regime occupancy across {len(samples)} windows",
                  color="white")
    plt.tight_layout()
    out_path = FIG_DIR / save_name
    plt.savefig(out_path, dpi=120, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    return out_path
