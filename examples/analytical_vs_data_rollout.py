#!/usr/bin/env python3
"""Evaluate analytical model against BeamNG data (one-step, teacher-forced).

Frame conventions: world frame (X, Y, yaw), body frame (vx, vy, r).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terrain_aware_model import vehicle_dynamics

# ── Utilities ─────────────────────────────────────────────────────────────


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def _infer_steer_sign(rule: str) -> float:
    s = "".join(rule.split()).lower()
    return -1.0 if "delta_cmd=-steering_input" in s else 1.0


# ── Config loading ────────────────────────────────────────────────────────


def load_model_config(config_path: Path):
    """Return (params, terrain_cfg, steer_sign) from a YAML config."""
    import yaml  # type: ignore

    cfg = yaml.safe_load(config_path.read_text("utf-8")) or {}
    terrain = {"dhx_default": 0.0, "dhy_default": 0.0, "mu_effective": 1.0}
    steer_sign = 1.0

    nominal = {k: float(v) for k, v in cfg.get("nominal_params", {}).items()}
    for k, v in cfg.get("optional_params", {}).items():
        if isinstance(v, (int, float)):
            nominal[k] = float(v)

    for k in terrain:
        if k in cfg.get("terrain_params", {}):
            terrain[k] = float(cfg["terrain_params"][k])

    rule = (cfg.get("model", {}).get("sign_convention", {})
               .get("steering_to_delta_cmd", ""))
    if rule:
        steer_sign = _infer_steer_sign(rule)

    return nominal, terrain, steer_sign


# ── Dynamics helpers ──────────────────────────────────────────────────────


def build_constant_terrain(cfg: dict[str, float]):
    dhx = float(cfg.get("dhx_default", 0.0))
    dhy = float(cfg.get("dhy_default", 0.0))
    mu = float(cfg.get("mu_effective", 1.0))
    return lambda _x, _y: (dhx, dhy, mu, None)


def rk4_step(x, u, dt, p, terrain_fn=None):
    def f(s):
        return vehicle_dynamics(0.0, s, u, p, terrain=terrain_fn)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def map_controls(steer, throttle, p, steer_sign):
    return np.array([steer_sign * float(steer) * p["delta_max"],
                     np.clip(float(throttle), 0.0, 1.0) * p["a_max"]])


def build_state8(x, y, yaw, vx, vy, r, steer, throttle, p, steer_sign, mode):
    if mode == "mapped":
        delta = steer_sign * float(steer) * p["delta_max"]
        a = np.clip(float(throttle), 0.0, 1.0) * p["a_max"]
    else:
        delta = a = 0.0
    return np.array([x, y, yaw, vx, vy, r, delta, a], dtype=float)


# ── Data loading ──────────────────────────────────────────────────────────


def load_raw_aligned(data_root: Path, transmission: str, terrain: str):
    base = data_root / "data" / transmission / "train" / terrain
    odom = np.genfromtxt(base / "odom.csv", delimiter=",", names=True)
    ctrl = np.genfromtxt(base / "controls.csv", delimiter=",", names=True)

    n = min(len(odom), len(ctrl))
    if n < 30:
        raise ValueError("Not enough aligned samples.")
    odom, ctrl = odom[:n], ctrl[:n]

    return {
        "time": odom["time"],
        "x": odom["pos_x"], "y": odom["pos_y"],
        "yaw": quat_to_yaw(odom["quat_x"], odom["quat_y"],
                            odom["quat_z"], odom["quat_w"]),
        "vx": odom["vel_x"], "vy": odom["vel_y"], "r": odom["ang_vel_z"],
        "steer": ctrl["steering_input"], "throttle": ctrl["throttle_input"],
    }


def select_split(data, split, train_ratio, downsample):
    n = len(data["time"])
    cut = int(np.clip(train_ratio, 0.05, 0.95) * n)
    sl = slice(0, cut) if split == "train" else slice(cut, n)
    out = {k: v[sl] for k, v in data.items()}
    if downsample > 1:
        out = {k: v[::downsample] for k, v in out.items()}
    if len(out["time"]) < 30:
        raise ValueError("Split too short after downsampling.")
    return out


def truncate_duration(data, max_seconds):
    if max_seconds <= 0.0:
        return data
    keep = (data["time"] - data["time"][0]) <= max_seconds
    if int(np.sum(keep)) < 30:
        raise ValueError(f"Duration-limited segment too short ({int(np.sum(keep))} samples).")
    return {k: v[keep] for k, v in data.items()}


# ── Metrics ───────────────────────────────────────────────────────────────


def compute_metrics(prefix, pred, gt):
    valid = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(gt), axis=1)
    if not np.any(valid):
        raise ValueError("No valid samples for metric computation.")
    pred, gt = pred[valid], gt[valid]

    dx, dy = pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1]
    pos_err = np.hypot(dx, dy)
    yaw_err = wrap_to_pi(pred[:, 2] - gt[:, 2])
    return {
        f"{prefix}_rmse_pos_m": _rmse(pos_err),
        f"{prefix}_rmse_x_m": _rmse(dx), f"{prefix}_rmse_y_m": _rmse(dy),
        f"{prefix}_rmse_yaw_deg": float(np.degrees(_rmse(yaw_err))),
        f"{prefix}_rmse_vx_mps": _rmse(pred[:, 3] - gt[:, 3]),
        f"{prefix}_rmse_vy_mps": _rmse(pred[:, 4] - gt[:, 4]),
        f"{prefix}_rmse_r_radps": _rmse(pred[:, 5] - gt[:, 5]),
        f"{prefix}_mae_pos_m": float(np.mean(pos_err)),
        f"{prefix}_mae_yaw_deg": float(np.degrees(np.mean(np.abs(yaw_err)))),
    }


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_state_tracking(out_dir, t, gt, one_step):
    labels = [("X [m]", 0), ("Y [m]", 1), ("Yaw [deg]", 2),
              ("vx [m/s]", 3), ("vy [m/s]", 4), ("r [rad/s]", 5)]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    flat = axes.ravel()

    for i, (ylabel, idx) in enumerate(labels):
        ax = flat[i]
        g = np.degrees(gt[:, idx]) if idx == 2 else gt[:, idx]
        o = np.degrees(one_step[:, idx]) if idx == 2 else one_step[:, idx]
        ax.plot(t, g, "k--", lw=1.6, label="BeamNG")
        ax.plot(t, o, color="tab:blue", lw=1.0, alpha=0.85, label="One-step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if i >= 4:
            ax.set_xlabel("Time [s]")

    handles, lbls = flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "state_tracking.png", dpi=220)
    plt.close(fig)


def plot_errors(out_dir, t, gt, one_step):
    pos_err = np.linalg.norm(one_step[:, :2] - gt[:, :2], axis=1)
    yaw_err = np.degrees(np.abs(wrap_to_pi(one_step[:, 2] - gt[:, 2])))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, data, ylabel, label in [
        (axes[0], pos_err, "Position error [m]", "One-step"),
        (axes[1], yaw_err, "Yaw abs error [deg]", "One-step"),
        (axes[2], one_step[:, 3] - gt[:, 3], "vx error [m/s]", "One-step vx err"),
    ]:
        ax.plot(t, data, color="tab:blue", alpha=0.85, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "errors.png", dpi=220)
    plt.close(fig)


def plot_path(out_dir, gt, one_step):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(gt[:, 0], gt[:, 1], "k--", lw=2.0, label="BeamNG trajectory")
    ax.plot(one_step[:, 0], one_step[:, 1], color="tab:blue", lw=1.3, label="One-step")
    ax.scatter(gt[0, 0], gt[0, 1], s=50, color="tab:green", label="Start")
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="X [m]", ylabel="Y [m]", title="Analytical Model vs BeamNG")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "path_prediction_vs_data.png", dpi=220)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Analytical model vs BeamNG trajectory data.")
    a = p.add_argument
    a("--data-root", type=Path, default=root / "function_encoder_beamng")
    a("--transmission", default="m1")
    a("--terrain", default="asphalt")
    a("--split", choices=["train", "test"], default="test")
    a("--train-ratio", type=float, default=0.8)
    a("--downsample", type=int, default=1)
    a("--max-test-seconds", type=float, default=250.0,
      help="Cap test-split duration (<=0 disables).")
    a("--max-steps", type=int, default=0, help="0 => full split.")
    a("--max-dt-factor", type=float, default=3.0,
      help="Skip transitions with dt > factor * median_dt.")
    a("--steer-sign", type=float, default=None, choices=[-1.0, 1.0])
    a("--internal-init", choices=["mapped", "zero"], default="mapped")
    a("--config", type=Path,
      default=root / "configs" / "analytical_scintilla_asphalt.yaml")
    a("--output-dir", type=Path,
      default=root / "results" / "analytical_vs_data")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Config & data
    p, terrain_cfg, steer_sign_cfg = load_model_config(args.config)
    steer_sign = steer_sign_cfg if args.steer_sign is None else float(args.steer_sign)
    terrain_fn = build_constant_terrain(terrain_cfg)

    seg = select_split(
        load_raw_aligned(args.data_root, args.transmission, args.terrain),
        args.split, args.train_ratio, max(1, args.downsample))
    if args.split == "test" and args.max_test_seconds > 0:
        seg = truncate_duration(seg, args.max_test_seconds)

    t = seg["time"]
    dt_raw = np.diff(t)
    dt_pos = dt_raw[dt_raw > 1e-6]
    if dt_pos.size == 0:
        raise ValueError("No positive dt values in selected split.")
    dt_median = float(np.median(dt_pos))
    dt = np.where(dt_raw > 1e-6, dt_raw, dt_median)

    n_steps = len(t) - 1
    if args.max_steps > 0:
        n_steps = min(n_steps, args.max_steps)

    gt6 = np.column_stack([seg[k][:n_steps + 1]
                           for k in ("x", "y", "yaw", "vx", "vy", "r")])
    valid_mask = dt[:n_steps] <= args.max_dt_factor * dt_median

    # One-step (teacher-forced) predictions
    one_step8 = np.full((n_steps + 1, 8), np.nan)
    one_step8[0] = build_state8(
        seg["x"][0], seg["y"][0], seg["yaw"][0],
        seg["vx"][0], seg["vy"][0], seg["r"][0],
        seg["steer"][0], seg["throttle"][0],
        p, steer_sign, args.internal_init)

    for k in range(n_steps):
        xk = build_state8(
            seg["x"][k], seg["y"][k], seg["yaw"][k],
            seg["vx"][k], seg["vy"][k], seg["r"][k],
            seg["steer"][k], seg["throttle"][k],
            p, steer_sign, args.internal_init)
        if valid_mask[k]:
            uk = map_controls(seg["steer"][k], seg["throttle"][k], p, steer_sign)
            one_step8[k + 1] = rk4_step(xk, uk, float(dt[k]), p, terrain_fn)

    one_step6 = one_step8[:, :6]

    # Metrics & output
    metrics = {
        "split": args.split, "n_steps": n_steps,
        "dt_median": dt_median,
        "dt_max_used": float(np.max(dt[:n_steps][valid_mask])) if valid_mask.any() else float("nan"),
        "dt_outlier_threshold": args.max_dt_factor * dt_median,
        "n_valid_steps": int(valid_mask.sum()),
        "n_skipped_dt_outliers": int((~valid_mask).sum()),
        "downsample": max(1, args.downsample),
        "max_test_seconds": args.max_test_seconds,
        "segment_duration_seconds": float(t[n_steps] - t[0]),
        "max_dt_factor": args.max_dt_factor,
        "steer_sign": steer_sign, "internal_init": args.internal_init,
        "terrain_mu_effective": terrain_cfg.get("mu_effective", 1.0),
        "terrain_dhx_default": terrain_cfg.get("dhx_default", 0.0),
        "terrain_dhy_default": terrain_cfg.get("dhy_default", 0.0),
        "model_params_used": {k: float(v) for k, v in p.items()},
    }
    metrics.update(compute_metrics("one_step", one_step6, gt6))

    t_rel = t[:n_steps + 1] - t[0]
    plot_path(args.output_dir, gt6, one_step6)
    plot_state_tracking(args.output_dir, t_rel, gt6, one_step6)
    plot_errors(args.output_dir, t_rel, gt6, one_step6)

    json_path = args.output_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nDone. Output: {args.output_dir}")
    print(f"  Metrics: {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
