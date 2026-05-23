"""Procedural test video generator pour benchmark comparison.

Crée 6 vidéos synthétiques exercising différents régimes dynamiques :
- bouncing_balls : Newtonian collisions (chaotic 2-body)
- rotating_shapes : periodic 2π cycle
- reaction_diffusion : Turing patterns (spatiotemporal)
- game_of_life : Conway cellular automata (discrete)
- double_pendulum_render : visual rendering of chaotic 4D system
- color_morph : smooth continuous color transitions

Chaque vidéo = signature dynamique connue → validation pipeline reproductible.
Génération via numpy + cv2 + ffmpeg, sans deps externes lourdes.
"""
from __future__ import annotations
import math
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import cv2


def _write_video_from_frames(
    frames: list[np.ndarray], path: Path, fps: int = 15,
) -> None:
    """Encode liste frames RGB uint8 → MP4 via ffmpeg pipe."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def gen_bouncing_balls(
    n_frames: int = 120, size: int = 256, n_balls: int = 5, seed: int = 0,
) -> list[np.ndarray]:
    """Newtonian bouncing balls dans box. 2-body collisions = chaos."""
    rng = np.random.RandomState(seed)
    positions = rng.uniform(20, size - 20, (n_balls, 2))
    velocities = rng.uniform(-3, 3, (n_balls, 2))
    radii = rng.randint(8, 16, n_balls)
    colors = rng.randint(50, 255, (n_balls, 3))
    frames = []
    for t in range(n_frames):
        positions += velocities
        # Bounce off walls
        for i in range(n_balls):
            for d in range(2):
                if positions[i, d] - radii[i] < 0:
                    positions[i, d] = radii[i]
                    velocities[i, d] = abs(velocities[i, d])
                elif positions[i, d] + radii[i] > size:
                    positions[i, d] = size - radii[i]
                    velocities[i, d] = -abs(velocities[i, d])
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(n_balls):
            cv2.circle(frame, tuple(positions[i].astype(int)),
                       int(radii[i]), tuple(int(c) for c in colors[i]), -1)
        frames.append(frame)
    return frames


def gen_rotating_shapes(
    n_frames: int = 120, size: int = 256,
) -> list[np.ndarray]:
    """Triangle + carré + cercle qui tournent. Periodic 2π."""
    frames = []
    cx, cy = size // 2, size // 2
    R = size // 3
    for t in range(n_frames):
        angle = 2 * math.pi * t / n_frames
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        # Triangle
        tri_pts = np.array([
            [cx + int(R * math.cos(angle + a)), cy + int(R * math.sin(angle + a))]
            for a in [0, 2 * math.pi / 3, 4 * math.pi / 3]
        ])
        cv2.fillPoly(frame, [tri_pts], (255, 100, 200))
        # Square inner
        s = R // 2
        sq_angle = -angle
        sq_pts = np.array([
            [cx + int(s * math.cos(sq_angle + a)), cy + int(s * math.sin(sq_angle + a))]
            for a in [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]
        ])
        cv2.fillPoly(frame, [sq_pts], (100, 200, 255))
        # Inner circle dot orbiting
        ox = cx + int(R * 0.4 * math.cos(angle * 3))
        oy = cy + int(R * 0.4 * math.sin(angle * 3))
        cv2.circle(frame, (ox, oy), 10, (255, 255, 100), -1)
        frames.append(frame)
    return frames


def gen_reaction_diffusion(
    n_frames: int = 120, size: int = 128, dt: float = 1.0,
    Du: float = 0.16, Dv: float = 0.08, f_param: float = 0.035, k_param: float = 0.060,
) -> list[np.ndarray]:
    """Gray-Scott reaction-diffusion (Turing pattern). Spatiotemporal complexity."""
    u = np.ones((size, size), dtype=np.float32)
    v = np.zeros((size, size), dtype=np.float32)
    # Seed perturbation center
    r = size // 8
    cx, cy = size // 2, size // 2
    u[cx-r:cx+r, cy-r:cy+r] = 0.5
    v[cx-r:cx+r, cy-r:cy+r] = 0.25

    frames = []
    steps_per_frame = 8
    for t in range(n_frames):
        for _ in range(steps_per_frame):
            # 5-point laplacian
            lap_u = (
                -4 * u + np.roll(u, 1, 0) + np.roll(u, -1, 0)
                + np.roll(u, 1, 1) + np.roll(u, -1, 1)
            )
            lap_v = (
                -4 * v + np.roll(v, 1, 0) + np.roll(v, -1, 0)
                + np.roll(v, 1, 1) + np.roll(v, -1, 1)
            )
            uvv = u * v * v
            u += dt * (Du * lap_u - uvv + f_param * (1 - u))
            v += dt * (Dv * lap_v + uvv - (f_param + k_param) * v)
        # Render v channel colorized
        norm = np.clip(v * 4.0, 0, 1)
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[..., 0] = (norm * 255).astype(np.uint8)
        rgb[..., 1] = (norm * 100).astype(np.uint8)
        rgb[..., 2] = ((1 - norm) * 200).astype(np.uint8)
        # Upscale for visibility
        rgb_up = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        frames.append(rgb_up)
    return frames


def gen_game_of_life(
    n_frames: int = 120, size: int = 80, seed: int = 42,
) -> list[np.ndarray]:
    """Conway's Game of Life. Discrete cellular automaton, complex emergence."""
    rng = np.random.RandomState(seed)
    grid = (rng.random((size, size)) < 0.3).astype(np.uint8)
    frames = []
    for t in range(n_frames):
        # Render
        rgb = np.stack([grid * 100, grid * 220, grid * 200], axis=-1).astype(np.uint8)
        rgb_up = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        frames.append(rgb_up)
        # Step
        neighbors = sum(
            np.roll(np.roll(grid, dx, 0), dy, 1)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)
        )
        new = ((grid == 1) & ((neighbors == 2) | (neighbors == 3))) | (
            (grid == 0) & (neighbors == 3)
        )
        grid = new.astype(np.uint8)
    return frames


def gen_double_pendulum_render(
    n_frames: int = 120, size: int = 256, dt: float = 0.04,
) -> list[np.ndarray]:
    """Double pendulum rendered visually. 4D Hamiltonian chaos."""
    l1, l2, m1, m2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    th1, th2 = math.pi * 0.85, math.pi * 0.5
    w1, w2 = 0.0, 0.0

    def deriv(th1, th2, w1, w2):
        d = 2 * m1 + m2 - m2 * math.cos(2 * th1 - 2 * th2)
        dw1 = (
            -g * (2 * m1 + m2) * math.sin(th1)
            - m2 * g * math.sin(th1 - 2 * th2)
            - 2 * math.sin(th1 - th2) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * math.cos(th1 - th2))
        ) / (l1 * d)
        dw2 = (
            2 * math.sin(th1 - th2) * (
                w1 * w1 * l1 * (m1 + m2)
                + g * (m1 + m2) * math.cos(th1)
                + w2 * w2 * l2 * m2 * math.cos(th1 - th2)
            )
        ) / (l2 * d)
        return w1, w2, dw1, dw2

    cx, cy = size // 2, size // 3
    scale = size // 5
    frames = []
    trail = []
    for t in range(n_frames):
        # RK4
        k1 = deriv(th1, th2, w1, w2)
        k2 = deriv(th1 + 0.5 * dt * k1[0], th2 + 0.5 * dt * k1[1], w1 + 0.5 * dt * k1[2], w2 + 0.5 * dt * k1[3])
        k3 = deriv(th1 + 0.5 * dt * k2[0], th2 + 0.5 * dt * k2[1], w1 + 0.5 * dt * k2[2], w2 + 0.5 * dt * k2[3])
        k4 = deriv(th1 + dt * k3[0], th2 + dt * k3[1], w1 + dt * k3[2], w2 + dt * k3[3])
        th1 += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        th2 += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        w1 += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        w2 += dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6

        x1 = cx + int(scale * math.sin(th1))
        y1 = cy + int(scale * math.cos(th1))
        x2 = x1 + int(scale * math.sin(th2))
        y2 = y1 + int(scale * math.cos(th2))
        trail.append((x2, y2))
        if len(trail) > 30:
            trail.pop(0)

        frame = np.zeros((size, size, 3), dtype=np.uint8)
        # Trail (fade)
        for i, (tx, ty) in enumerate(trail):
            alpha = i / len(trail)
            cv2.circle(frame, (tx, ty), 2, (int(255 * alpha), int(100 * alpha), int(200 * alpha)), -1)
        # Rods + masses
        cv2.line(frame, (cx, cy), (x1, y1), (180, 180, 180), 2)
        cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x1, y1), 8, (100, 200, 255), -1)
        cv2.circle(frame, (x2, y2), 8, (255, 100, 200), -1)
        frames.append(frame)
    return frames


def gen_color_morph(
    n_frames: int = 120, size: int = 256,
) -> list[np.ndarray]:
    """Smooth continuous color gradient morphing. Test ultra-smooth signal."""
    frames = []
    for t in range(n_frames):
        phase = 2 * math.pi * t / n_frames
        # Gradient horizontal + vertical avec phase shift
        y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
        r = (np.sin(x * 4 + phase) + 1) * 127
        g = (np.sin(y * 4 + phase * 1.3) + 1) * 127
        b = (np.sin((x + y) * 3 + phase * 0.7) + 1) * 127
        frame = np.stack([r, g, b], axis=-1).astype(np.uint8)
        frames.append(frame)
    return frames


GENERATORS = {
    "bouncing_balls": gen_bouncing_balls,
    "rotating_shapes": gen_rotating_shapes,
    "reaction_diffusion": gen_reaction_diffusion,
    "game_of_life": gen_game_of_life,
    "double_pendulum_render": gen_double_pendulum_render,
    "color_morph": gen_color_morph,
}


def generate_all(out_dir: Path, fps: int = 15) -> dict:
    """Génère 6 vidéos synthétiques canoniques + metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in GENERATORS.items():
        path = out_dir / f"{name}.mp4"
        if path.exists():
            results[name] = {"path": str(path), "status": "cached"}
            continue
        frames = fn()
        _write_video_from_frames(frames, path, fps=fps)
        results[name] = {
            "path": str(path),
            "n_frames": len(frames),
            "size_kb": path.stat().st_size // 1024 if path.exists() else 0,
            "status": "generated",
        }
    return results
