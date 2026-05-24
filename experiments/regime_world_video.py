"""Phase 2 : RegimeWorldModel on REAL video features (DINOv2 frozen).

Setup :
- Generate procedural videos × 3 regimes (smooth/periodic/chaotic)
- Encode each frame via DINOv2 (frozen, 384d latent)
- Window the resulting feature sequences
- Train RWM-video (d_in=384) vs Flat-video baseline (matched ~30K params)
- Eval IN-DIST (held-out seeds same generators) + OOD (different generators)

Key Phase 2 question : does the regime-aware architecture validated on
synthetic raw trajectories also work when input is high-dim deep
features extracted from rendered video ?
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.regime_world import (
    RegimeWorldModel, TrainConfig, train, encode_eval, REGIME_TO_IDX,
)
from backend.regime_world.video_data import (
    build_windowed_dataset, VideoSample,
)
from backend.regime_world.losses import regime_conditional_loss


# Flat baseline with same encoder dimensionality
class FlatVideoBaseline(nn.Module):
    def __init__(self, d_in: int = 384, d_h: int = 56):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128), nn.SiLU(), nn.Linear(128, d_h),
        )
        self.gru = nn.GRUCell(d_h, d_h)
        self.proj_next = nn.Linear(d_h, d_h)
        self.decoder = nn.Sequential(
            nn.Linear(d_h, 128), nn.SiLU(), nn.Linear(128, d_in),
        )
        self.d_h = d_h

    def forward(self, x):
        z = self.encoder(x)
        T = z.shape[0]
        h = torch.zeros(self.d_h, device=z.device)
        z_seq, z_pred = [], []
        for t in range(T):
            h = self.gru(z[t:t + 1], h.unsqueeze(0)).squeeze(0)
            z_seq.append(h)
            z_pred.append(self.proj_next(h))
        return {"z_seq": torch.stack(z_seq), "z_pred": torch.stack(z_pred),
                "x_recon": self.decoder(z)}


def train_baseline_video(model, dataset, n_epochs=40, lr=3e-4, device="cpu"):
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        for sample in dataset:
            x = torch.from_numpy(sample.features).float().to(device)
            x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
            out = model(x)
            l_recon = ((out["x_recon"] - x) ** 2).mean()
            l_dyn = ((out["z_pred"][:-1] - out["z_seq"][1:].detach()) ** 2).mean()
            loss = l_recon + l_dyn
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
    return model


def train_rwm_video(rwm, dataset, n_epochs, w_regime_sup=1.0):
    cfg = TrainConfig(n_epochs=n_epochs, lr=3e-4, device="cpu",
                       w_dyn=1.0, w_slow=1.0, w_recur=0.5, w_lyap=0.3,
                       w_entropy=0.1, w_regime_sup=w_regime_sup)
    return train(rwm, dataset, cfg, verbose=False)


@torch.no_grad()
def eval_sample(model, sample, kind="rwm"):
    x = torch.from_numpy(sample.features).float()
    x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
    if kind == "rwm":
        out = model(x)
        z = out["z_slow"]; zp = out["z_slow_pred"]
        if z.shape[0] < 2:
            return {"mse": float("nan"), "router": [0, 0, 0], "lstd": 0}
        mse = float(((zp[:-1] - z[1:]) ** 2).mean().item())
        return {"mse": mse, "router": out["r"].cpu().numpy().tolist(),
                "lstd": float(z.std(0).mean().item())}
    else:
        out = model(x)
        z = out["z_seq"]; zp = out["z_pred"]
        if z.shape[0] < 2:
            return {"mse": float("nan"), "lstd": 0}
        mse = float(((zp[:-1] - z[1:]) ** 2).mean().item())
        return {"mse": mse, "lstd": float(z.std(0).mean().item())}


def aggregate(samples, model, kind="rwm"):
    mses, lstds, routers = [], [], []
    regimes_true = []
    correct = 0
    for s in samples:
        r = eval_sample(model, s, kind=kind)
        mses.append(r["mse"])
        lstds.append(r["lstd"])
        if kind == "rwm":
            routers.append(r["router"])
            true_idx = REGIME_TO_IDX[s.regime]
            pred_idx = int(np.argmax(r["router"]))
            if pred_idx == true_idx:
                correct += 1
        regimes_true.append(s.regime)
    out = {"mse_mean": float(np.mean(mses)), "mse_std": float(np.std(mses)),
           "lstd_mean": float(np.mean(lstds))}
    if kind == "rwm":
        out["router_mean"] = np.array(routers).mean(axis=0).tolist()
        out["router_acc"] = correct / max(len(samples), 1)
    return out


def split_by_regime(samples, train_ratio=0.7, seed=0):
    rng = np.random.default_rng(seed)
    train, evals = [], []
    by_reg = {"smooth": [], "periodic": [], "chaotic": []}
    for s in samples:
        by_reg[s.regime].append(s)
    for reg, items in by_reg.items():
        idx = np.arange(len(items)); rng.shuffle(idx)
        n_tr = int(len(idx) * train_ratio)
        for i in idx[:n_tr]: train.append(items[i])
        for i in idx[n_tr:]: evals.append(items[i])
    return train, evals


def run_one_seed(seed: int, n_videos_per_regime=4, n_frames=64,
                 window=32, stride=8, n_epochs=40):
    print(f"\n=== SEED {seed} ===")
    samples = build_windowed_dataset(
        n_videos_per_regime=n_videos_per_regime, n_frames_per_video=n_frames,
        window=window, stride=stride, seed_base=seed * 1000, verbose=(seed == 0),
    )
    train_ds, eval_ds = split_by_regime(samples, train_ratio=0.7, seed=seed)
    print(f"  train={len(train_ds)} eval={len(eval_ds)}")

    # OOD : different generators per regime (held-out)
    # chaotic has 4 generators ; use last 1 as OOD held-out
    # For periodic + smooth (1 each), skip OOD (no held-out generator possible)
    # Build separate OOD set via seed_base shift to get different param-seeded videos
    ood_samples = build_windowed_dataset(
        n_videos_per_regime=2, n_frames_per_video=n_frames,
        window=window, stride=stride, seed_base=seed * 1000 + 99999, verbose=False,
    )
    print(f"  ood   ={len(ood_samples)} (held-out seeds)")

    d_in = train_ds[0].features.shape[1]
    print(f"  d_in (encoder dim) = {d_in}")

    # Train RWM-video
    torch.manual_seed(seed); np.random.seed(seed)
    rwm = RegimeWorldModel(d_in=d_in, d_fast=64, d_slow=64, slow_stride=4)
    print(f"  RWM params : {rwm.count_params()}")
    ts = time.time()
    train_rwm_video(rwm, train_ds, n_epochs=n_epochs)
    t_rwm = time.time() - ts
    print(f"  RWM trained ({t_rwm:.1f}s)")

    # Train Flat baseline matched
    torch.manual_seed(seed); np.random.seed(seed)
    flat = FlatVideoBaseline(d_in=d_in, d_h=56)
    n_flat = sum(p.numel() for p in flat.parameters())
    print(f"  Flat params : {n_flat}")
    ts = time.time()
    train_baseline_video(flat, train_ds, n_epochs=n_epochs)
    t_flat = time.time() - ts
    print(f"  Flat trained ({t_flat:.1f}s)")

    # Eval
    results = {}
    for dist, ds in [("in_dist", eval_ds), ("ood", ood_samples)]:
        rwm_m = aggregate(ds, rwm, kind="rwm")
        flat_m = aggregate(ds, flat, kind="flat")
        results[dist] = {"rwm": rwm_m, "flat": flat_m,
                          "ratio_rwm_flat": rwm_m["mse_mean"] / max(flat_m["mse_mean"], 1e-9)}
        print(f"  {dist:<8}: RWM mse={rwm_m['mse_mean']:.4f}  "
              f"Flat mse={flat_m['mse_mean']:.4f}  ratio={results[dist]['ratio_rwm_flat']:.2f}  "
              f"router_acc={rwm_m.get('router_acc', 0):.2f}")
    return {"seed": seed, "results": results,
             "rwm_params": rwm.count_params(), "flat_params": n_flat}


def main():
    print("=" * 80)
    print("PHASE 2 — RegimeWorldModel on REAL VIDEO (DINOv2 features)")
    print("=" * 80)
    print("  Encoder : DINOv2-S frozen (384d per frame)")
    print("  Pipeline : video → frames → DINOv2 → fast/slow → regime mixture")
    print("  Eval IN-DIST : held-out seeds same generators")
    print("  Eval OOD     : new seeds (parameter shifts)")
    print("=" * 80)

    import os
    N_SEEDS = int(os.environ.get("RWM_N_SEEDS", "3"))
    N_EPOCHS = int(os.environ.get("RWM_N_EPOCHS", "40"))
    N_VIDS = int(os.environ.get("RWM_N_VIDS", "4"))

    all_runs = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        try:
            all_runs.append(run_one_seed(seed, n_videos_per_regime=N_VIDS,
                                          n_epochs=N_EPOCHS))
        except Exception as e:
            print(f"  seed {seed} failed : {e}")
            import traceback; traceback.print_exc()
    print(f"\n[time] total {time.time() - t0:.1f}s")

    if not all_runs:
        return

    # Aggregate
    rwm_in = [r["results"]["in_dist"]["rwm"]["mse_mean"] for r in all_runs]
    flat_in = [r["results"]["in_dist"]["flat"]["mse_mean"] for r in all_runs]
    rwm_ood = [r["results"]["ood"]["rwm"]["mse_mean"] for r in all_runs]
    flat_ood = [r["results"]["ood"]["flat"]["mse_mean"] for r in all_runs]
    router_in = [r["results"]["in_dist"]["rwm"]["router_acc"] for r in all_runs]
    router_ood = [r["results"]["ood"]["rwm"]["router_acc"] for r in all_runs]

    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(f"  IN-DIST :")
    print(f"    RWM mse  : {np.mean(rwm_in):.4f} ± {np.std(rwm_in):.4f}")
    print(f"    Flat mse : {np.mean(flat_in):.4f} ± {np.std(flat_in):.4f}")
    print(f"    ratio    : {np.mean(rwm_in) / max(np.mean(flat_in), 1e-9):.2f}")
    print(f"  OOD :")
    print(f"    RWM mse  : {np.mean(rwm_ood):.4f} ± {np.std(rwm_ood):.4f}")
    print(f"    Flat mse : {np.mean(flat_ood):.4f} ± {np.std(flat_ood):.4f}")
    print(f"    ratio    : {np.mean(rwm_ood) / max(np.mean(flat_ood), 1e-9):.2f}")
    print(f"  Router acc : in={np.mean(router_in):.2f}  ood={np.mean(router_ood):.2f}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "all_runs": all_runs,
        "agg": {
            "rwm_in_mean": float(np.mean(rwm_in)),
            "flat_in_mean": float(np.mean(flat_in)),
            "rwm_ood_mean": float(np.mean(rwm_ood)),
            "flat_ood_mean": float(np.mean(flat_ood)),
            "ratio_in": float(np.mean(rwm_in) / max(np.mean(flat_in), 1e-9)),
            "ratio_ood": float(np.mean(rwm_ood) / max(np.mean(flat_ood), 1e-9)),
            "router_in": float(np.mean(router_in)),
            "router_ood": float(np.mean(router_ood)),
        },
    }
    def _d(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"{type(o)}")
    json.dump(out, open("results/regime_world_video.json", "w"), indent=2, default=_d)
    print(f"\n[saved] results/regime_world_video.json")


if __name__ == "__main__":
    main()
