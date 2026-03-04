#!/usr/bin/env python3
"""D1 — State-by-state error accumulation profile.

Runs 5s rollouts and records per-timestep error for each state
(vx, vy, yaw, x, y, r) at every integration step.
Reports median + Q90 error curves to show which state diverges first.

Uses TEST split with tuned (best) config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.analytical_open_loop_rollouts import (
    build_constant_terrain,
    build_state8,
    load_model_config,
    load_raw_aligned,
    map_controls,
    rk4_step,
    select_split,
    truncate_duration,
    wrap_to_pi,
)

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="D1: error accumulation profile")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs" / "analytical_scintilla_asphalt_best.yaml")
    p.add_argument("--data-root", type=Path,
                   default=ROOT / "function_encoder_beamng")
    p.add_argument("--transmission", default="m1")
    p.add_argument("--terrain", default="asphalt")
    p.add_argument("--split", default="test")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-seconds", type=float, default=250.0)
    p.add_argument("--rollout-seconds", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-rollouts", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "diag_d1_error_accumulation")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p, terrain_cfg, steer_sign = load_model_config(args.config)
    terrain_fn = build_constant_terrain(terrain_cfg)

    raw = load_raw_aligned(args.data_root, args.transmission, args.terrain)
    seg = select_split(raw, args.split, args.train_ratio, 1)
    if args.max_seconds > 0:
        seg = truncate_duration(seg, args.max_seconds)

    n_steps = int(round(args.rollout_seconds / args.dt))
    n_data = len(seg["time"])
    max_start = n_data - 1 - n_steps
    if max_start < 0:
        raise ValueError("Not enough data for rollout horizon.")

    rng = np.random.default_rng(args.seed)
    n_roll = min(args.n_rollouts, max_start + 1)
    starts = rng.choice(np.arange(max_start + 1), size=n_roll, replace=False)

    # Storage: (n_rollouts, n_steps+1) for each error type
    time_axis = np.arange(n_steps + 1) * args.dt
    err_x = np.zeros((n_roll, n_steps + 1))
    err_y = np.zeros((n_roll, n_steps + 1))
    err_pos = np.zeros((n_roll, n_steps + 1))
    err_yaw = np.zeros((n_roll, n_steps + 1))
    err_vx = np.zeros((n_roll, n_steps + 1))
    err_vy = np.zeros((n_roll, n_steps + 1))
    err_r = np.zeros((n_roll, n_steps + 1))

    # Also record initial conditions for regime tagging
    init_vx = np.zeros(n_roll)
    init_throttle = np.zeros(n_roll)
    init_steer = np.zeros(n_roll)

    for ri, si in enumerate(starts):
        si = int(si)
        xs = build_state8(
            seg["x"][si], seg["y"][si], seg["yaw"][si],
            seg["vx"][si], seg["vy"][si], seg["r"][si],
            seg["steer"][si], seg["throttle"][si],
            p, steer_sign, "mapped")

        gt = np.column_stack([
            seg["x"][si:si + n_steps + 1],
            seg["y"][si:si + n_steps + 1],
            seg["yaw"][si:si + n_steps + 1],
            seg["vx"][si:si + n_steps + 1],
            seg["vy"][si:si + n_steps + 1],
            seg["r"][si:si + n_steps + 1],
        ])

        pred = np.zeros((n_steps + 1, 8))
        pred[0] = xs
        for k in range(n_steps):
            u = map_controls(seg["steer"][si + k], seg["throttle"][si + k],
                             p, steer_sign)
            pred[k + 1] = rk4_step(pred[k], u, args.dt, p, terrain_fn)

        err_x[ri] = pred[:, 0] - gt[:, 0]
        err_y[ri] = pred[:, 1] - gt[:, 1]
        err_pos[ri] = np.hypot(err_x[ri], err_y[ri])
        err_yaw[ri] = np.degrees(np.abs(wrap_to_pi(pred[:, 2] - gt[:, 2])))
        err_vx[ri] = pred[:, 3] - gt[:, 3]
        err_vy[ri] = pred[:, 4] - gt[:, 4]
        err_r[ri] = pred[:, 5] - gt[:, 5]

        init_vx[ri] = float(seg["vx"][si])
        init_throttle[ri] = float(seg["throttle"][si])
        init_steer[ri] = float(np.abs(seg["steer"][si]))

    # ── Compute statistics ───────────────────────────────────────────────
    def stats(arr_abs):
        return {
            "median": np.median(arr_abs, axis=0),
            "q90": np.percentile(arr_abs, 90, axis=0),
            "q10": np.percentile(arr_abs, 10, axis=0),
            "mean": np.mean(arr_abs, axis=0),
        }

    pos_stats = stats(err_pos)
    yaw_stats = stats(err_yaw)
    vx_stats = stats(np.abs(err_vx))
    vy_stats = stats(np.abs(err_vy))
    r_stats = stats(np.abs(np.degrees(err_r)))

    # ── Print key numbers ────────────────────────────────────────────────
    print("=" * 70)
    print(f"D1 — ERROR ACCUMULATION PROFILE ({args.split} split, {n_roll} rollouts)")
    print(f"Config: {args.config.name}")
    print("=" * 70)

    for horizon_s in [0.5, 1.0, 2.0, 3.0, 5.0]:
        idx = min(int(round(horizon_s / args.dt)), n_steps)
        print(f"\n--- At t={horizon_s:.1f}s (step {idx}) ---")
        print(f"  Position err: median={pos_stats['median'][idx]:.3f}m, "
              f"Q90={pos_stats['q90'][idx]:.3f}m")
        print(f"  Yaw err:      median={yaw_stats['median'][idx]:.2f}deg, "
              f"Q90={yaw_stats['q90'][idx]:.2f}deg")
        print(f"  |vx| err:     median={vx_stats['median'][idx]:.3f}m/s, "
              f"Q90={vx_stats['q90'][idx]:.3f}m/s")
        print(f"  |vy| err:     median={vy_stats['median'][idx]:.3f}m/s, "
              f"Q90={vy_stats['q90'][idx]:.3f}m/s")
        print(f"  |r| err:      median={r_stats['median'][idx]:.3f}deg/s, "
              f"Q90={r_stats['q90'][idx]:.3f}deg/s")

    # ── Plot 1: Error growth curves ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    def plot_band(ax, t, s, label, color, ylabel):
        ax.plot(t, s["median"], color=color, lw=2, label=f"median")
        ax.fill_between(t, s["q10"], s["q90"], alpha=0.2, color=color,
                        label="Q10–Q90")
        ax.plot(t, s["mean"], color=color, lw=1, ls="--", alpha=0.6,
                label="mean")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plot_band(axes[0, 0], time_axis, pos_stats, "Position error", "tab:red", "[m]")
    plot_band(axes[0, 1], time_axis, yaw_stats, "Yaw error", "tab:blue", "[deg]")
    plot_band(axes[0, 2], time_axis, vx_stats, "|vx| error", "tab:green", "[m/s]")
    plot_band(axes[1, 0], time_axis, vy_stats, "|vy| error", "tab:orange", "[m/s]")
    plot_band(axes[1, 1], time_axis, r_stats, "|yaw rate| error", "tab:purple", "[deg/s]")

    # Plot 6: position error split by speed regime
    ax6 = axes[1, 2]
    speed_med = np.median(init_vx)
    slow_mask = init_vx < speed_med
    fast_mask = ~slow_mask
    if np.any(slow_mask):
        ax6.plot(time_axis, np.median(err_pos[slow_mask], axis=0),
                 "b-", lw=2, label=f"slow (<{speed_med:.1f} m/s)")
    if np.any(fast_mask):
        ax6.plot(time_axis, np.median(err_pos[fast_mask], axis=0),
                 "r-", lw=2, label=f"fast (>={speed_med:.1f} m/s)")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("[m]")
    ax6.set_title("Position error by speed regime")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    fig.suptitle(f"D1: Error Accumulation ({args.split}, {n_roll} rollouts, config={args.config.name})",
                 fontsize=13)
    fig.savefig(args.output_dir / "d1_error_growth.png", dpi=180)
    plt.close(fig)

    # ── Plot 2: vx error (signed) to see bias direction ──────────────────
    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    vx_signed_median = np.median(err_vx, axis=0)
    vx_signed_q10 = np.percentile(err_vx, 10, axis=0)
    vx_signed_q90 = np.percentile(err_vx, 90, axis=0)
    ax_a.plot(time_axis, vx_signed_median, "g-", lw=2, label="median")
    ax_a.fill_between(time_axis, vx_signed_q10, vx_signed_q90, alpha=0.2,
                       color="green", label="Q10–Q90")
    ax_a.axhline(0, color="k", ls=":", alpha=0.3)
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel("vx error [m/s] (pred - GT)")
    ax_a.set_title("Signed vx error — shows bias direction")
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    yaw_signed = np.degrees(wrap_to_pi(np.zeros_like(err_yaw)))  # placeholder
    # Actually compute signed yaw error properly
    # err_yaw was stored as absolute, recompute signed
    yaw_signed_med = np.median(np.degrees(err_r), axis=0)  # r error as proxy
    ax_b.plot(time_axis, np.median(err_vx, axis=0), "g-", lw=2, label="vx err median")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(time_axis, np.median(err_pos, axis=0), "r-", lw=2, label="pos err median")
    ax_b.set_xlabel("Time [s]")
    ax_b.set_ylabel("vx error [m/s]", color="g")
    ax_b2.set_ylabel("pos error [m]", color="r")
    ax_b.set_title("vx error vs position error — causal link?")
    ax_b.legend(loc="upper left", fontsize=8)
    ax_b2.legend(loc="upper right", fontsize=8)
    ax_b.grid(alpha=0.3)

    fig2.suptitle("D1: Signed vx error and position error", fontsize=13)
    fig2.savefig(args.output_dir / "d1_vx_bias.png", dpi=180)
    plt.close(fig2)

    # ── Save JSON ────────────────────────────────────────────────────────
    def to_list(arr):
        return [float(x) for x in arr]

    payload = {
        "config": str(args.config),
        "split": args.split,
        "n_rollouts": n_roll,
        "rollout_seconds": args.rollout_seconds,
        "dt": args.dt,
        "n_steps": n_steps,
        "time_axis": to_list(time_axis),
        "position_error": {
            "median": to_list(pos_stats["median"]),
            "q90": to_list(pos_stats["q90"]),
            "mean": to_list(pos_stats["mean"]),
        },
        "yaw_error_deg": {
            "median": to_list(yaw_stats["median"]),
            "q90": to_list(yaw_stats["q90"]),
            "mean": to_list(yaw_stats["mean"]),
        },
        "vx_error_abs": {
            "median": to_list(vx_stats["median"]),
            "q90": to_list(vx_stats["q90"]),
            "mean": to_list(vx_stats["mean"]),
        },
        "vx_error_signed_median": to_list(np.median(err_vx, axis=0)),
    }
    (args.output_dir / "d1_results.json").write_text(json.dumps(payload, indent=2))

    print(f"\nPlots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
