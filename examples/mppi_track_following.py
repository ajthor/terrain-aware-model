"""Sample-based MPPI path-following example on a closed track."""
from __future__ import annotations

import math

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from tqdm import tqdm

from terrain_aware_model import vehicle_dynamics


P = {
    "m": 11.0,
    "Iz": 0.25,
    "lf": 0.24,
    "lr": 0.24,
    "L": 0.48,
    "h_com": 0.12,
    "g": 9.81,
    "Cf0": 8e4,
    "Cr0": 9e4,
    "k_a": 1.5e3,
    "tau_delta": 0.06,
    "tau_a": 0.12,
    "delta_max": 0.55,
    "a_min": -8.0,
    "a_max": 8.0,
}


def build_track(num_points=400):
    straight_len = 36.0
    radius = 10.0
    half_straight = straight_len * 0.5
    arc_points = num_points // 4
    straight_points = num_points // 2

    theta_right = np.linspace(-0.5 * math.pi, 0.5 * math.pi, arc_points, endpoint=False)
    theta_left = np.linspace(0.5 * math.pi, 1.5 * math.pi, arc_points, endpoint=False)

    right_arc = np.column_stack((
        half_straight + radius * np.cos(theta_right),
        radius * np.sin(theta_right),
    ))
    left_arc = np.column_stack((
        -half_straight + radius * np.cos(theta_left),
        radius * np.sin(theta_left),
    ))

    top = np.column_stack((
        np.linspace(half_straight, -half_straight, straight_points // 2, endpoint=False),
        np.full(straight_points // 2, radius),
    ))
    bottom = np.column_stack((
        np.linspace(-half_straight, half_straight, straight_points // 2, endpoint=False),
        np.full(straight_points // 2, -radius),
    ))

    return np.vstack((right_arc, top, left_arc, bottom))


def cross_track_error_sq(points, position):
    deltas = points - position[None, :]
    distances = np.einsum("ij,ij->i", deltas, deltas)
    idx = int(np.argmin(distances))
    p0 = points[idx]
    p1 = points[(idx + 1) % len(points)]
    seg = p1 - p0
    seg_len2 = seg[0] * seg[0] + seg[1] * seg[1]
    if seg_len2 < 1e-9:
        return float(distances[idx])
    t = np.clip(((position - p0) @ seg) / seg_len2, 0.0, 1.0)
    proj = p0 + t * seg
    err = position - proj
    return float(err @ err)


def cross_track_error_sq_batch(points, positions):
    errors = np.empty(positions.shape[0], dtype=float)
    for i, pos in enumerate(positions):
        errors[i] = cross_track_error_sq(points, pos)
    return errors


def rk4_step(x, u, dt, p):
    k1 = vehicle_dynamics(0.0, x, u, p)
    k2 = vehicle_dynamics(0.0, x + 0.5 * dt * k1, u, p)
    k3 = vehicle_dynamics(0.0, x + 0.5 * dt * k2, u, p)
    k4 = vehicle_dynamics(0.0, x + dt * k3, u, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def clamp_controls(controls, p):
    controls = np.array(controls, copy=True)
    controls[..., 0] = np.clip(controls[..., 0], p["a_min"], p["a_max"])
    controls[..., 1] = np.clip(
        controls[..., 1], -p["delta_max"], p["delta_max"])
    return controls


def rollout_cost_batch(x0, controls, dt, p, track):
    num_samples, horizon, _ = controls.shape
    x = np.broadcast_to(x0, (num_samples, x0.shape[-1])).copy()
    costs = np.zeros(num_samples, dtype=float)
    prev_accel = np.zeros(num_samples, dtype=float)
    prev_steer = np.zeros(num_samples, dtype=float)

    for step in range(horizon):
        accel = controls[:, step, 0]
        steer = controls[:, step, 1]
        u = np.stack((steer, accel), axis=1)
        x = rk4_step(x, u, dt, p)

        costs += 4.0 * cross_track_error_sq_batch(track, x[:, :2])

        speed_error = x[:, 3] - 8.0
        costs += 0.5 * speed_error * speed_error

        costs += 0.2 * accel * accel + 0.6 * steer * steer
        costs += 0.1 * (accel - prev_accel) ** 2 + \
            0.2 * (steer - prev_steer) ** 2
        prev_accel = accel
        prev_steer = steer

    return costs


def rollout_states_batch(x0, controls, dt, p):
    num_samples, horizon, _ = controls.shape
    x = np.broadcast_to(x0, (num_samples, x0.shape[-1])).copy()
    positions = np.empty((num_samples, horizon + 1, 2), dtype=float)
    positions[:, 0, :] = x[:, :2]

    for step in range(horizon):
        accel = controls[:, step, 0]
        steer = controls[:, step, 1]
        u = np.stack((steer, accel), axis=1)
        x = rk4_step(x, u, dt, p)
        positions[:, step + 1, :] = x[:, :2]

    return positions


track = build_track()
x0 = np.array([track[0, 0], track[0, 1], 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
horizon = 20
dt = 0.1
num_samples = 100
lambda_ = 3.0
noise_std = (2.0, 0.25)
viz_samples = 20
max_trail = 100

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(track[:, 0], track[:, 1], "k--", label="Track")
trail = LineCollection([], linewidths=2.0, label="Vehicle")
ax.add_collection(trail)
airplane = Polygon(np.zeros((4, 2)), closed=True, facecolor="tab:red", edgecolor="tab:red", alpha=0.9)
ax.add_patch(airplane)
candidate_lines = [
    ax.plot([], [], color="tab:blue", alpha=0.2, linewidth=1.0)[0]
    for _ in range(viz_samples)
]
ax.axis("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("MPPI Path Following")
ax.legend()
plt.tight_layout()

xs = []
ys = []
x = x0.copy()
control_seq = np.zeros((horizon, 2), dtype=float)
rng = np.random.default_rng(7)
step_index = 0
last_tick = time.perf_counter()

with tqdm(total=None, desc="MPPI", unit="step") as pbar:
    while True:
        noise = rng.normal(scale=noise_std, size=(num_samples, horizon, 2))
        candidates = clamp_controls(control_seq + noise, P)
        costs = rollout_cost_batch(x, candidates, dt, P, track)

        costs = costs - np.min(costs)
        weights = np.exp(-costs / max(lambda_, 1e-6))
        weights /= np.sum(weights)
        control_seq = np.tensordot(weights, candidates, axes=(0, 0))

        accel, steer = control_seq[0]
        u = np.array([steer, accel])
        x = rk4_step(x, u, dt, P)
        control_seq = np.vstack([control_seq[1:], control_seq[-1]])

        xs.append(x[0])
        ys.append(x[1])
        if len(xs) > max_trail:
            xs = xs[-max_trail:]
            ys = ys[-max_trail:]
        if len(xs) > 1:
            points = np.column_stack((xs, ys))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            alphas = np.linspace(0.05, 1.0, len(segments))
            colors = np.zeros((len(segments), 4), dtype=float)
            colors[:, 0] = 1.0
            colors[:, 3] = alphas
            trail.set_segments(segments)
            trail.set_color(colors)
        heading_length = 1.2
        wing_span = 0.9
        nose = np.array([heading_length, 0.0])
        left = np.array([-0.4 * heading_length, 0.5 * wing_span])
        tail = np.array([-0.2 * heading_length, 0.0])
        right = np.array([-0.4 * heading_length, -0.5 * wing_span])
        shape = np.stack((nose, left, tail, right))
        cpsi = np.cos(x[2])
        spsi = np.sin(x[2])
        rot = np.array([[cpsi, -spsi], [spsi, cpsi]])
        verts = (shape @ rot.T) + x[:2]
        airplane.set_xy(verts)

        show_count = min(viz_samples, num_samples)
        cand_positions = rollout_states_batch(
            x, candidates[:show_count], dt, P)
        for idx, line in enumerate(candidate_lines):
            if idx < show_count:
                line.set_data(
                    cand_positions[idx, :, 0], cand_positions[idx, :, 1])
            else:
                line.set_data([], [])

        plt.pause(0.001)
        now = time.perf_counter()
        elapsed = now - last_tick
        if elapsed < dt:
            time.sleep(dt - elapsed)
        last_tick = time.perf_counter()
        step_index += 1
        pbar.update(1)

plt.ioff()
plt.show()
