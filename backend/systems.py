"""Zoo de systemes dynamiques canoniques pour benchmark + demo.
Chaque generateur retourne (norm_coords [-1,1], raw_coords, dt, var_names, meta)."""
from __future__ import annotations
import numpy as np


def _normalise(raw: np.ndarray) -> np.ndarray:
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    return (2 * (raw - mins) / span - 1).astype(np.float32)


def lorenz(n: int = 2500, dt: float = 0.01,
           sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> dict:
    x, y, z = 0.1, 0.0, 0.0
    raw = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
        raw[i] = [x, y, z]
    return {
        "name": "Lorenz",
        "coords": _normalise(raw).tolist(), "raw_coords": raw.tolist(),
        "dt": dt, "var_names": ["x", "y", "z"],
        "meta": {"sigma": sigma, "rho": rho, "beta": beta},
        "type": "chaotic 3D ODE",
    }


def rossler(n: int = 2500, dt: float = 0.05,
            a: float = 0.2, b: float = 0.2, c: float = 5.7) -> dict:
    x, y, z = 1.0, 1.0, 1.0
    raw = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt; y += dy * dt; z += dz * dt
        raw[i] = [x, y, z]
    return {
        "name": "Rossler",
        "coords": _normalise(raw).tolist(), "raw_coords": raw.tolist(),
        "dt": dt, "var_names": ["x", "y", "z"],
        "meta": {"a": a, "b": b, "c": c},
        "type": "chaotic 3D ODE · single funnel",
    }


def van_der_pol(n: int = 2500, dt: float = 0.05, mu: float = 2.0) -> dict:
    """Oscillateur limit-cycle. 2D embed via Takens delay pour visu 3D."""
    x, y = 0.5, 0.0
    raw = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        dx = y
        dy = mu * (1 - x * x) * y - x
        x += dx * dt; y += dy * dt
        raw[i] = [x, y]
    # Embed Takens (x, y, x_delayed) pour 3D
    tau = 12
    if n > tau:
        z_col = np.zeros(n, dtype=np.float32)
        z_col[tau:] = raw[:-tau, 0]
        raw3 = np.stack([raw[:, 0], raw[:, 1], z_col], axis=1)
    else:
        raw3 = np.concatenate([raw, np.zeros((n, 1))], axis=1)
    return {
        "name": "Van der Pol",
        "coords": _normalise(raw3).tolist(), "raw_coords": raw3.tolist(),
        "dt": dt, "var_names": ["x", "y", "x_τ"],
        "meta": {"mu": mu, "delay": tau},
        "type": "2D limit cycle · Takens-embedded",
    }


def henon(n: int = 4000, a: float = 1.4, b: float = 0.3) -> dict:
    """Map discrete chaotique 2D + Takens 3D."""
    x, y = 0.0, 0.0
    raw = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        x_new = 1 - a * x * x + y
        y_new = b * x
        x, y = x_new, y_new
        raw[i] = [x, y]
    # Delay embed pour 3D
    tau = 3
    z_col = np.zeros(n, dtype=np.float32)
    z_col[tau:] = raw[:-tau, 0]
    raw3 = np.stack([raw[:, 0], raw[:, 1], z_col], axis=1)
    return {
        "name": "Hénon",
        "coords": _normalise(raw3).tolist(), "raw_coords": raw3.tolist(),
        "dt": 1.0, "var_names": ["x", "y", "x_τ"],
        "meta": {"a": a, "b": b, "delay": tau},
        "type": "discrete 2D map · strange attractor",
    }


def logistic_map(n: int = 4000, r: float = 3.9, embed_dim: int = 3, tau: int = 1) -> dict:
    """Map 1D, sortie embedded en 3D pour visualiser bifurcation."""
    x = 0.4
    series = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = r * x * (1 - x)
        series[i] = x
    # Takens 3D
    span = (embed_dim - 1) * tau
    raw3 = np.zeros((n - span, embed_dim), dtype=np.float32)
    for j in range(embed_dim):
        raw3[:, j] = series[j * tau : j * tau + (n - span)]
    return {
        "name": "Logistic",
        "coords": _normalise(raw3).tolist(), "raw_coords": raw3.tolist(),
        "dt": 1.0, "var_names": [f"x_{i}τ" for i in range(embed_dim)],
        "meta": {"r": r, "embed_dim": embed_dim, "delay": tau},
        "type": "1D map · embedded · period-doubling",
    }


def double_pendulum(n: int = 3000, dt: float = 0.02,
                    l1: float = 1.0, l2: float = 1.0,
                    m1: float = 1.0, m2: float = 1.0, g: float = 9.81,
                    th1_0: float = np.pi * 0.8, th2_0: float = np.pi * 0.6) -> dict:
    """RK4 sur double pendule. Sortie : (theta1, theta2, omega1) en 3D."""
    th1, th2, w1, w2 = th1_0, th2_0, 0.0, 0.0

    def deriv(th1, th2, w1, w2):
        d = 2 * m1 + m2 - m2 * np.cos(2 * th1 - 2 * th2)
        dw1 = (-g * (2 * m1 + m2) * np.sin(th1)
               - m2 * g * np.sin(th1 - 2 * th2)
               - 2 * np.sin(th1 - th2) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * np.cos(th1 - th2))) / (l1 * d)
        dw2 = (2 * np.sin(th1 - th2) * (w1 * w1 * l1 * (m1 + m2)
               + g * (m1 + m2) * np.cos(th1)
               + w2 * w2 * l2 * m2 * np.cos(th1 - th2))) / (l2 * d)
        return w1, w2, dw1, dw2

    raw = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        k1 = deriv(th1, th2, w1, w2)
        k2 = deriv(th1 + 0.5 * dt * k1[0], th2 + 0.5 * dt * k1[1],
                   w1 + 0.5 * dt * k1[2], w2 + 0.5 * dt * k1[3])
        k3 = deriv(th1 + 0.5 * dt * k2[0], th2 + 0.5 * dt * k2[1],
                   w1 + 0.5 * dt * k2[2], w2 + 0.5 * dt * k2[3])
        k4 = deriv(th1 + dt * k3[0], th2 + dt * k3[1],
                   w1 + dt * k3[2], w2 + dt * k3[3])
        th1 += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        th2 += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        w1 += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        w2 += dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
        raw[i] = [th1, th2, w1]
    return {
        "name": "Double Pendulum",
        "coords": _normalise(raw).tolist(), "raw_coords": raw.tolist(),
        "dt": dt, "var_names": ["θ₁", "θ₂", "ω₁"],
        "meta": {"l1": l1, "l2": l2, "g": g},
        "type": "Hamiltonian chaos · 4D phase space",
    }


SYSTEMS = {
    "lorenz": lorenz,
    "rossler": rossler,
    "van_der_pol": van_der_pol,
    "henon": henon,
    "logistic": logistic_map,
    "double_pendulum": double_pendulum,
}


def list_systems() -> list[dict]:
    """Metadata pour UI picker (instancie chacun en mini-batch pour recuperer le 'type')."""
    return [
        {"id": "lorenz", "name": "Lorenz", "type": "chaotic 3D ODE"},
        {"id": "rossler", "name": "Rössler", "type": "chaotic 3D ODE · funnel"},
        {"id": "van_der_pol", "name": "Van der Pol", "type": "limit cycle"},
        {"id": "henon", "name": "Hénon", "type": "discrete 2D map"},
        {"id": "logistic", "name": "Logistic", "type": "1D map · period-doubling"},
        {"id": "double_pendulum", "name": "Double Pendulum", "type": "Hamiltonian chaos"},
    ]
