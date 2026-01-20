"""Classic MPC path-following example on a closed track."""
from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from scipy.optimize import minimize
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

    theta_right = np.linspace(-0.5 * math.pi, 0.5 *
                              math.pi, arc_points, endpoint=False)
    theta_left = np.linspace(0.5 * math.pi, 1.5 *
                             math.pi, arc_points, endpoint=False)

    right_arc = np.column_stack((
        half_straight + radius * np.cos(theta_right),
        radius * np.sin(theta_right),
    ))
    left_arc = np.column_stack((
        -half_straight + radius * np.cos(theta_left),
        radius * np.sin(theta_left),
    ))

    top = np.column_stack((
        np.linspace(half_straight, -half_straight,
                    straight_points // 2, endpoint=False),
        np.full(straight_points // 2, radius),
    ))
    bottom = np.column_stack((
        np.linspace(-half_straight, half_straight,
                    straight_points // 2, endpoint=False),
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


def rk4_step(x, u, dt, p):
    k1 = vehicle_dynamics(0.0, x, u, p)
    k2 = vehicle_dynamics(0.0, x + 0.5 * dt * k1, u, p)
    k3 = vehicle_dynamics(0.0, x + 0.5 * dt * k2, u, p)
    k4 = vehicle_dynamics(0.0, x + dt * k3, u, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_cost(x0, controls, dt, p, track):
    x = np.array(x0, dtype=float)
    cost = 0.0
    prev_accel = 0.0
    prev_steer = 0.0

    for accel, steer in controls:
        u = np.array([steer, accel])
        x = rk4_step(x, u, dt, p)

        cost += 4.0 * cross_track_error_sq(track, x[:2])

        speed_error = x[3] - 8.0
        cost += 0.5 * speed_error * speed_error

        cost += 0.2 * accel * accel + 0.6 * steer * steer
        cost += 0.1 * (accel - prev_accel) ** 2 + \
            0.2 * (steer - prev_steer) ** 2
        prev_accel = accel
        prev_steer = steer

    return float(cost)


track = build_track()
x0 = np.array([track[0, 0], track[0, 1], 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
horizon = 8
dt = 0.2
max_trail = 300

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(track[:, 0], track[:, 1], "k--", label="Track")
trail = LineCollection([], linewidths=2.0, label="Vehicle")
ax.add_collection(trail)
airplane = Polygon(np.zeros((4, 2)), closed=True,
                   facecolor="tab:red", edgecolor="tab:red", alpha=0.9)
ax.add_patch(airplane)
ax.axis("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("MPC Path Following")
ax.legend()
plt.tight_layout()

xs = []
ys = []
x = x0.copy()
control_seq = np.zeros((horizon, 2), dtype=float)
last_tick = time.perf_counter()

with tqdm(total=None, desc="MPC", unit="step") as pbar:
    while True:
        def cost_fn(flat_controls):
            controls = flat_controls.reshape(horizon, 2)
            return rollout_cost(x, controls, dt, P, track)

        result = minimize(
            cost_fn,
            control_seq.reshape(-1),
            method="L-BFGS-B",
            bounds=[(P["a_min"], P["a_max"]),
                    (-P["delta_max"], P["delta_max"])] * horizon,
            options={"maxiter": 40, "ftol": 1e-2},
        )
        control_seq = result.x.reshape(horizon, 2)

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

        plt.pause(0.001)
        now = time.perf_counter()
        elapsed = now - last_tick
        if elapsed < dt:
            time.sleep(dt - elapsed)
        last_tick = time.perf_counter()
        pbar.update(1)

plt.ioff()
plt.show()
