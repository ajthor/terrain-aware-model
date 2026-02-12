#!/usr/bin/env python3
"""Evaluate analytical model with short open-loop control-replay rollouts.

Each rollout starts from ground-truth state, replays logged controls for a
fixed horizon, integrates with fixed dt, and runs without feedback.
"""
from __future__ import annotations

import argparse
import csv
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


def compute_rollout_metrics(pred, gt):
    dx, dy = pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1]
    pos_err = np.hypot(dx, dy)
    yaw_err = wrap_to_pi(pred[:, 2] - gt[:, 2])
    return {
        "rmse_pos_m": _rmse(pos_err),
        "rmse_x_m": _rmse(dx), "rmse_y_m": _rmse(dy),
        "rmse_yaw_deg": float(np.degrees(_rmse(yaw_err))),
        "rmse_vx_mps": _rmse(pred[:, 3] - gt[:, 3]),
        "rmse_vy_mps": _rmse(pred[:, 4] - gt[:, 4]),
        "rmse_r_radps": _rmse(pred[:, 5] - gt[:, 5]),
        "mae_pos_m": float(np.mean(pos_err)),
        "mae_yaw_deg": float(np.degrees(np.mean(np.abs(yaw_err)))),
        "final_pos_err_m": float(pos_err[-1]),
        "final_yaw_err_deg": float(np.degrees(np.abs(yaw_err[-1]))),
    }


def summarize_metrics(items):
    if not items:
        return {}
    skip = {"rollout_id", "plot_id", "start_idx", "start_time"}
    out = {}
    for k in (k for k in items[0] if k not in skip):
        a = np.array([d[k] for d in items])
        out[k] = {"mean": float(a.mean()), "std": float(a.std()),
                   "min": float(a.min()), "max": float(a.max())}
    return out


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_rollouts_on_path(out_path, gt_chunks, pred_chunks,
                          start_indices, start_times):
    n = len(pred_chunks)
    if n == 0:
        return
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))
    cell = 4.4

    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr),
                             squeeze=False, constrained_layout=True)
    flat = axes.ravel()

    for i, (gt, pr, si, st) in enumerate(
            zip(gt_chunks, pred_chunks, start_indices, start_times)):
        ax = flat[i]
        ax.plot(gt[:, 0], gt[:, 1], "--k", lw=1.2, alpha=0.9, label="GT")
        ax.plot(pr[:, 0], pr[:, 1], color="tab:red", lw=1.5, label="Pred")
        ax.scatter(*pr[0], color="tab:red", s=14)
        ax.scatter(*gt[0], color="k", s=10)

        all_xy = np.vstack([gt, pr])
        lo, hi = all_xy.min(0), all_xy.max(0)
        half = 0.58 * max((hi - lo).max(), 1e-6)
        mid = 0.5 * (lo + hi)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="X [m]", ylabel="Y [m]",
               title=f"R{i+1} | idx={si} | t={st:.2f}s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(n, len(flat)):
        flat[j].axis("off")

    fig.suptitle("Open-loop rollout chunks: GT vs Prediction", fontsize=12)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Open-loop analytical rollouts vs BeamNG logs.")
    a = p.add_argument
    a("--data-root", type=Path, default=root / "function_encoder_beamng")
    a("--transmission", default="m1")
    a("--terrain", default="asphalt")
    a("--split", choices=["train", "test"], default="test")
    a("--train-ratio", type=float, default=0.8)
    a("--downsample", type=int, default=1)
    a("--max-test-seconds", type=float, default=250.0,
      help="Cap test-split duration (<=0 disables).")
    a("--rollout-seconds", type=float, default=0.5)
    a("--dt", type=float, default=0.05, help="Integration step [s].")
    a("--n-rollouts", type=int, default=9, help="Rollouts per plot.")
    a("--n-plots", type=int, default=3)
    a("--seed", type=int, default=7)
    a("--steer-sign", type=float, default=None, choices=[-1.0, 1.0])
    a("--internal-init", choices=["mapped", "zero"], default="mapped")
    a("--config", type=Path,
      default=root / "configs" / "analytical_scintilla_asphalt.yaml")
    a("--output-dir", type=Path,
      default=root / "results" / "analytical_open_loop_rollouts")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.rollout_seconds <= 0 or args.dt <= 0:
        raise ValueError("--rollout-seconds and --dt must be > 0")
    if args.n_rollouts <= 0 or args.n_plots <= 0:
        raise ValueError("--n-rollouts and --n-plots must be > 0")

    # Config & data
    p, terrain_cfg, steer_sign_cfg = load_model_config(args.config)
    steer_sign = steer_sign_cfg if args.steer_sign is None else float(args.steer_sign)
    terrain_fn = build_constant_terrain(terrain_cfg)

    seg = select_split(
        load_raw_aligned(args.data_root, args.transmission, args.terrain),
        args.split, args.train_ratio, max(1, args.downsample))
    if args.split == "test" and args.max_test_seconds > 0:
        seg = truncate_duration(seg, args.max_test_seconds)

    # Rollout geometry
    n_steps = int(round(args.rollout_seconds / args.dt))
    if n_steps < 2:
        raise ValueError("Rollout horizon too short.")
    max_start = len(seg["time"]) - 1 - n_steps
    if max_start < 0:
        raise ValueError("Not enough data for requested rollout horizon.")

    total = args.n_rollouts * args.n_plots
    valid = np.arange(max_start + 1)
    if total > len(valid):
        raise ValueError(
            f"Requested {total} rollouts but only {len(valid)} valid starts.")

    rng = np.random.default_rng(args.seed)
    starts = rng.choice(valid, size=total, replace=False)
    gt6_full = np.column_stack([seg[k] for k in ("x", "y", "yaw", "vx", "vy", "r")])

    # Execute rollouts
    all_metrics: list[dict] = []
    plot_data: list[list[tuple]] = [[] for _ in range(args.n_plots)]

    for rid, si in enumerate(starts):
        pred = np.zeros((n_steps + 1, 8))
        pred[0] = build_state8(
            seg["x"][si], seg["y"][si], seg["yaw"][si],
            seg["vx"][si], seg["vy"][si], seg["r"][si],
            seg["steer"][si], seg["throttle"][si],
            p, steer_sign, args.internal_init)

        for k in range(n_steps):
            u = map_controls(seg["steer"][si + k], seg["throttle"][si + k],
                             p, steer_sign)
            pred[k + 1] = rk4_step(pred[k], u, args.dt, p, terrain_fn)

        gt6 = gt6_full[si : si + n_steps + 1]
        m = compute_rollout_metrics(pred[:, :6], gt6)
        m.update(rollout_id=rid, plot_id=rid // args.n_rollouts,
                 start_idx=int(si), start_time=float(seg["time"][si]))
        all_metrics.append(m)
        plot_data[rid // args.n_rollouts].append(
            (gt6[:, :2], pred[:, :2], int(si), float(seg["time"][si])))

    # Write plots
    for pid, entries in enumerate(plot_data):
        if not entries:
            continue
        gt_xy, pr_xy, idxs, times = zip(*entries)
        plot_rollouts_on_path(
            args.output_dir / f"path_rollouts_plot_{pid + 1:02d}.png",
            list(gt_xy), list(pr_xy), list(idxs), list(times))

    # Write CSV
    csv_path = args.output_dir / "rollout_metrics.csv"
    if all_metrics:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
            w.writeheader()
            w.writerows(all_metrics)

    # Write JSON
    json_path = args.output_dir / "metrics.json"
    payload = {
        "settings": {
            "split": args.split, "train_ratio": args.train_ratio,
            "downsample": max(1, args.downsample),
            "max_test_seconds": args.max_test_seconds,
            "segment_duration_seconds": float(seg["time"][-1] - seg["time"][0]),
            "rollout_seconds": args.rollout_seconds, "dt": args.dt,
            "rollout_steps": n_steps,
            "n_rollouts_per_plot": args.n_rollouts, "n_plots": args.n_plots,
            "total_rollouts": total, "seed": args.seed,
            "steer_sign": steer_sign, "internal_init": args.internal_init,
            "terrain_mu_effective": terrain_cfg.get("mu_effective", 1.0),
            "terrain_dhx_default": terrain_cfg.get("dhx_default", 0.0),
            "terrain_dhy_default": terrain_cfg.get("dhy_default", 0.0),
        },
        "model_params_used": {k: float(v) for k, v in p.items()},
        "summary_metrics": summarize_metrics(all_metrics),
        "per_rollout_metrics": all_metrics,
        "plot_files": [f"path_rollouts_plot_{i + 1:02d}.png"
                       for i in range(args.n_plots)],
        "metrics_csv": csv_path.name,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"\nDone. Output: {args.output_dir}")
    print(f"  JSON: {json_path.name}  CSV: {csv_path.name}"
          f"  Plots: {args.n_plots} x {args.n_rollouts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
