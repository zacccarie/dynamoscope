"""Synthetic dynamical systems zoo — raw trajectories + regime labels + ground-truth λ.

Génère trajectoires 1D/multi-dim brutes (pas de rendering vidéo) pour entraîner
le RegimeWorldModel directement sur dynamique. Chaque sample inclut :
- trajectory (T, d)
- regime label : "smooth" | "periodic" | "chaotic"
- lambda_true (per-sample, ground truth de la littérature)

Systèmes implémentés :
    smooth    : damped oscillator, exponential decay
    periodic  : sin, Van der Pol (cycle limite), Kuramoto sync regime
    chaotic   : Lorenz, Rossler, Logistic r=4, Henon
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class TrajSample:
    """Sample d'entraînement : trajectoire raw + label régime + λ ground truth."""
    traj: np.ndarray              # (T, d)
    regime: str                   # "smooth" | "periodic" | "chaotic"
    lambda_true: float            # per-sample (per natural time unit)
    system_name: str
    params: dict


def _rk4_step(f, s, dt):
    k1 = f(s)
    k2 = f(s + dt / 2 * k1)
    k3 = f(s + dt / 2 * k2)
    k4 = f(s + dt * k3)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================================
# SMOOTH systems
# ============================================================================

def gen_damped_oscillator(T: int = 200, gamma: float = 0.15, w0: float = 1.0,
                          seed: int = 0) -> TrajSample:
    rng = np.random.default_rng(seed)
    dt = 0.05
    f = lambda s: np.array([s[1], -2 * gamma * s[1] - w0 ** 2 * s[0]])
    s = np.array([1.0 + 0.1 * rng.standard_normal(), 0.0])
    traj = [s.copy()]
    for _ in range(T - 1):
        s = _rk4_step(f, s, dt)
        traj.append(s.copy())
    return TrajSample(
        traj=np.stack(traj),
        regime="smooth",
        lambda_true=-gamma,
        system_name="damped_osc",
        params={"gamma": gamma, "w0": w0},
    )


def gen_exp_decay(T: int = 200, tau: float = 5.0, seed: int = 0) -> TrajSample:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, T)
    x = np.exp(-t / tau) * (1.0 + 0.05 * rng.standard_normal(T))
    y = -np.exp(-t / tau) / tau * (1.0 + 0.05 * rng.standard_normal(T))
    traj = np.stack([x, y], axis=1)
    return TrajSample(
        traj=traj,
        regime="smooth",
        lambda_true=-1.0 / tau,
        system_name="exp_decay",
        params={"tau": tau},
    )


# ============================================================================
# PERIODIC systems
# ============================================================================

def gen_sine_pair(T: int = 200, freq: float = 0.05, seed: int = 0) -> TrajSample:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phase = 2 * np.pi * rng.uniform()
    x = np.sin(2 * np.pi * freq * t + phase)
    y = np.cos(2 * np.pi * freq * t + phase)
    traj = np.stack([x, y], axis=1)
    return TrajSample(
        traj=traj,
        regime="periodic",
        lambda_true=0.0,
        system_name="sine_pair",
        params={"freq": freq},
    )


def gen_van_der_pol(T: int = 200, mu: float = 1.5, seed: int = 0) -> TrajSample:
    """Van der Pol with μ > 0 : converges to limit cycle (periodic)."""
    rng = np.random.default_rng(seed)
    dt = 0.05
    f = lambda s: np.array([s[1], mu * (1 - s[0] ** 2) * s[1] - s[0]])
    s = np.array([0.5 + 0.1 * rng.standard_normal(), 0.0])
    # transient
    for _ in range(300):
        s = _rk4_step(f, s, dt)
    traj = [s.copy()]
    for _ in range(T - 1):
        s = _rk4_step(f, s, dt)
        traj.append(s.copy())
    return TrajSample(
        traj=np.stack(traj),
        regime="periodic",
        lambda_true=0.0,
        system_name="van_der_pol",
        params={"mu": mu},
    )


def gen_kuramoto_sync(T: int = 200, N: int = 8, K: float = 3.0,
                      seed: int = 0) -> TrajSample:
    """Kuramoto avec K > Kc : régime synchronisé (periodic)."""
    rng = np.random.default_rng(seed)
    omegas = 1.0 + 0.1 * rng.standard_normal(N)
    theta = 2 * np.pi * rng.uniform(size=N)
    dt = 0.05
    traj = []
    for _ in range(T):
        order = np.mean(np.exp(1j * theta))
        R = abs(order)
        psi = np.angle(order)
        dtheta = omegas + K * R * np.sin(psi - theta)
        theta = theta + dt * dtheta
        # Use order parameter + mean angle as 2D representation
        traj.append([R, psi])
    return TrajSample(
        traj=np.array(traj),
        regime="periodic",
        lambda_true=0.0,
        system_name="kuramoto",
        params={"N": N, "K": K},
    )


# ============================================================================
# CHAOTIC systems
# ============================================================================

def gen_lorenz(T: int = 200, sigma: float = 10.0, rho: float = 28.0,
               beta: float = 8 / 3, dt: float = 0.02, sub: int = 1,
               seed: int = 0) -> TrajSample:
    rng = np.random.default_rng(seed)
    f = lambda s: np.array([
        sigma * (s[1] - s[0]),
        s[0] * (rho - s[2]) - s[1],
        s[0] * s[1] - beta * s[2],
    ])
    s = np.array([1.0 + 0.1 * rng.standard_normal(), 1.0, 1.0])
    for _ in range(400):
        s = _rk4_step(f, s, dt)
    traj = []
    for _ in range(T):
        for _ in range(sub):
            s = _rk4_step(f, s, dt)
        traj.append(s.copy())
    return TrajSample(
        traj=np.stack(traj),
        regime="chaotic",
        lambda_true=0.906,  # per natural time unit
        system_name="lorenz",
        params={"sigma": sigma, "rho": rho, "beta": beta, "dt_eff": dt * sub},
    )


def gen_rossler(T: int = 200, a: float = 0.2, b: float = 0.2, c: float = 5.7,
                dt: float = 0.05, seed: int = 0) -> TrajSample:
    rng = np.random.default_rng(seed)
    f = lambda s: np.array([-s[1] - s[2], s[0] + a * s[1], b + s[2] * (s[0] - c)])
    s = np.array([1.0 + 0.1 * rng.standard_normal(), 1.0, 1.0])
    for _ in range(300):
        s = _rk4_step(f, s, dt)
    traj = []
    for _ in range(T):
        s = _rk4_step(f, s, dt)
        traj.append(s.copy())
    return TrajSample(
        traj=np.stack(traj),
        regime="chaotic",
        lambda_true=0.072,
        system_name="rossler",
        params={"a": a, "b": b, "c": c},
    )


def gen_logistic(T: int = 200, r: float = 4.0, seed: int = 0) -> TrajSample:
    """Logistic map at r=4 : full chaos. λ = ln 2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9)
    for _ in range(200):
        x = r * x * (1 - x)
    series = []
    for _ in range(T + 2):
        x = r * x * (1 - x)
        series.append(x)
    s = np.array(series)
    # Embed dim 3 via stacking
    traj = np.stack([s[:-2], s[1:-1], s[2:]], axis=1)
    return TrajSample(
        traj=traj,
        regime="chaotic",
        lambda_true=float(np.log(2.0)),
        system_name="logistic",
        params={"r": r},
    )


def gen_henon(T: int = 200, a: float = 1.4, b: float = 0.3,
              seed: int = 0) -> TrajSample:
    """Henon map. Strange attractor, λ ≈ 0.42."""
    rng = np.random.default_rng(seed)
    x, y = 0.1 + 0.01 * rng.standard_normal(), 0.0
    for _ in range(300):
        x, y = 1 - a * x * x + y, b * x
    traj = []
    for _ in range(T):
        x, y = 1 - a * x * x + y, b * x
        traj.append([x, y])
    return TrajSample(
        traj=np.array(traj),
        regime="chaotic",
        lambda_true=0.42,
        system_name="henon",
        params={"a": a, "b": b},
    )


# ============================================================================
# Registry + dataset builder
# ============================================================================

GENERATORS_PER_REGIME = {
    "smooth": [gen_damped_oscillator, gen_exp_decay],
    "periodic": [gen_sine_pair, gen_van_der_pol, gen_kuramoto_sync],
    "chaotic": [gen_lorenz, gen_rossler, gen_logistic, gen_henon],
}

REGIME_TO_IDX = {"smooth": 0, "periodic": 1, "chaotic": 2}


def make_dataset(n_per_regime: int = 24, T: int = 128, base_seed: int = 0
                 ) -> list[TrajSample]:
    """Génère N_per_regime samples par régime, balanced. Pad/crop à T points."""
    out = []
    seed = base_seed
    for regime, gens in GENERATORS_PER_REGIME.items():
        for i in range(n_per_regime):
            gen = gens[i % len(gens)]
            sample = gen(T=T, seed=seed)
            out.append(sample)
            seed += 1
    return out


def split_train_eval(dataset: list[TrajSample], train_ratio: float = 0.8,
                     seed: int = 0) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dataset))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    train = [dataset[i] for i in idx[:n_train]]
    eval_ = [dataset[i] for i in idx[n_train:]]
    return train, eval_
