"""Classic MPC path-following example on a closed track."""
from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
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


def terrain_height(x, y):
    return 0.6 * np.sin(0.12 * x) + 0.4 * np.cos(0.1 * y) + 0.2 * np.sin(0.06 * (x + y))


def terrain_gradient(x, y):
    dhx = 0.6 * 0.12 * np.cos(0.12 * x) + 0.2 * 0.06 * np.cos(0.06 * (x + y))
    dhy = -0.4 * 0.1 * np.sin(0.1 * y) + 0.2 * 0.06 * np.cos(0.06 * (x + y))
    return dhx, dhy


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
max_trail = 200
force_scale = 1.0
show_body_frame_forces = True

plt.ion()
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(4, 2, width_ratios=[3.2, 1.0], wspace=0.2, hspace=0.5)
ax = fig.add_subplot(gs[:, 0], projection="3d")
ax_fgx = fig.add_subplot(gs[0, 1])
ax_fgy = fig.add_subplot(gs[1, 1])
ax_vy = fig.add_subplot(gs[2, 1])
ax_vx = fig.add_subplot(gs[3, 1])
track_z = terrain_height(track[:, 0], track[:, 1])
x_min, x_max = track[:, 0].min() - 12.0, track[:, 0].max() + 12.0
y_min, y_max = track[:, 1].min() - 12.0, track[:, 1].max() + 12.0
grid_x = np.linspace(x_min, x_max, 60)
grid_y = np.linspace(y_min, y_max, 60)
mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
mesh_z = terrain_height(mesh_x, mesh_y)
ax.plot_surface(mesh_x, mesh_y, mesh_z, cmap="Greys", alpha=0.35, linewidth=0)
ax.plot(track[:, 0], track[:, 1], track_z, "k--", label="Track")
trail = Line3DCollection([], linewidths=2.0, label="Vehicle")
ax.add_collection(trail)
airplane = Poly3DCollection([np.zeros((4, 3))], facecolor="tab:red", edgecolor="tab:red", alpha=0.9)
ax.add_collection3d(airplane)
ax.set_box_aspect((1.8, 1.0, 0.5))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("MPC Path Following")
legend_handles = [
    Line2D([0], [0], color="k", linestyle="--", label="Track"),
    Line2D([0], [0], color="tab:red", linewidth=2.0, label="Vehicle trail"),
    Patch(facecolor="tab:red", edgecolor="tab:red", label="Vehicle"),
    Line2D([0], [0], color="tab:orange", linewidth=2.0, label="Fgx"),
    Line2D([0], [0], color="tab:green", linewidth=2.0, label="Fgy"),
]
ax.legend(handles=legend_handles, loc="upper left")
ax.view_init(elev=45, azim=-135)
ax.grid(False)
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.06)

fx_line, = ax_fgx.plot([], [], color="tab:orange")
fy_line, = ax_fgy.plot([], [], color="tab:green")
vy_line, = ax_vy.plot([], [], color="tab:purple")
vx_line, = ax_vx.plot([], [], color="tab:blue")
ax_fgx.set_ylabel("Fgx [N]")
ax_fgy.set_ylabel("Fgy [N]")
ax_vy.set_ylabel("vy [m/s]")
ax_vx.set_ylabel("vx [m/s]")
ax_vx.set_xlabel("time [s]")
for axis in (ax_fgx, ax_fgy, ax_vy, ax_vx):
    axis.grid(alpha=0.3)

xs = []
ys = []
times = []
fx_hist = []
fy_hist = []
vy_hist = []
vx_hist = []
x = x0.copy()
control_seq = np.zeros((horizon, 2), dtype=float)
last_tick = time.perf_counter()
force_quiver_x = None
force_quiver_y = None

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
        times.append(len(times) * dt)
        dhx, dhy = terrain_gradient(x[0], x[1])
        cpsi = np.cos(x[2])
        spsi = np.sin(x[2])
        s_slope = dhx * cpsi + dhy * spsi
        c_slope = -dhx * spsi + dhy * cpsi
        fx_hist.append(-P["m"] * P["g"] * s_slope)
        fy_hist.append(-P["m"] * P["g"] * c_slope)
        vy_hist.append(x[4])
        vx_hist.append(x[3])
        if len(xs) > max_trail:
            xs = xs[-max_trail:]
            ys = ys[-max_trail:]
            times = times[-max_trail:]
            fx_hist = fx_hist[-max_trail:]
            fy_hist = fy_hist[-max_trail:]
            vy_hist = vy_hist[-max_trail:]
            vx_hist = vx_hist[-max_trail:]
        if len(xs) > 1:
            z_vals = terrain_height(np.array(xs), np.array(ys))
            points = np.column_stack((xs, ys, z_vals))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            alphas = np.linspace(0.05, 1.0, len(segments))
            colors = np.zeros((len(segments), 4), dtype=float)
            colors[:, 0] = 0.84
            colors[:, 1] = 0.15
            colors[:, 2] = 0.16
            colors[:, 3] = alphas
            trail.set_segments(segments)
            trail.set_color(colors)
        heading_length = 2.4
        wing_span = 1.8
        nose = np.array([heading_length, 0.0])
        left = np.array([-0.4 * heading_length, 0.5 * wing_span])
        tail = np.array([-0.2 * heading_length, 0.0])
        right = np.array([-0.4 * heading_length, -0.5 * wing_span])
        shape = np.stack((nose, left, tail, right))
        cpsi = np.cos(x[2])
        spsi = np.sin(x[2])
        rot = np.array([[cpsi, -spsi], [spsi, cpsi]])
        verts2d = (shape @ rot.T) + x[:2]
        z_plane = terrain_height(x[0], x[1]) + 0.05
        verts = np.column_stack((verts2d, np.full(verts2d.shape[0], z_plane)))
        airplane.set_verts([verts])

        fx_body = fx_hist[-1]
        fy_body = fy_hist[-1]
        if show_body_frame_forces:
            fx_vec = np.array([fx_body * cpsi, fx_body * spsi])
            fy_vec = np.array([-fy_body * spsi, fy_body * cpsi])
        else:
            fx_vec = np.array([fx_body, 0.0])
            fy_vec = np.array([0.0, fy_body])
        base_z = terrain_height(x[0], x[1]) + 0.1
        if force_quiver_x is not None:
            force_quiver_x.remove()
        if force_quiver_y is not None:
            force_quiver_y.remove()
        force_quiver_x = ax.quiver(
            x[0],
            x[1],
            base_z,
            fx_vec[0] * force_scale,
            fx_vec[1] * force_scale,
            0.0,
            color="tab:orange",
            linewidth=2.0,
            arrow_length_ratio=0.04,
        )
        force_quiver_y = ax.quiver(
            x[0],
            x[1],
            base_z,
            fy_vec[0] * force_scale,
            fy_vec[1] * force_scale,
            0.0,
            color="tab:green",
            linewidth=2.0,
            arrow_length_ratio=0.04,
        )

        if len(times) > 1:
            fx_line.set_data(times, fx_hist)
            fy_line.set_data(times, fy_hist)
            vy_line.set_data(times, vy_hist)
            vx_line.set_data(times, vx_hist)
            ax_fgx.set_xlim(times[0], times[-1])
            ax_fgy.set_xlim(times[0], times[-1])
            ax_vy.set_xlim(times[0], times[-1])
            ax_vx.set_xlim(times[0], times[-1])
            fx_min, fx_max = min(fx_hist), max(fx_hist)
            fy_min, fy_max = min(fy_hist), max(fy_hist)
            vy_min, vy_max = min(vy_hist), max(vy_hist)
            vx_min, vx_max = min(vx_hist), max(vx_hist)
            ax_fgx.set_ylim(fx_min - 1.0, fx_max + 1.0)
            ax_fgy.set_ylim(fy_min - 1.0, fy_max + 1.0)
            ax_vy.set_ylim(vy_min - 0.5, vy_max + 0.5)
            ax_vx.set_ylim(vx_min - 0.5, vx_max + 0.5)

        plt.pause(0.001)
        now = time.perf_counter()
        elapsed = now - last_tick
        if elapsed < dt:
            time.sleep(dt - elapsed)
        last_tick = time.perf_counter()
        pbar.update(1)

plt.ioff()
plt.show()
