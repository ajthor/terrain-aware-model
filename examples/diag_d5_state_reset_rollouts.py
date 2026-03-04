#!/usr/bin/env python3
"""D5 — State-reset rollouts to isolate root cause of yaw divergence.

Runs rollouts while selectively resetting individual states to GT at each step.
By comparing which reset helps the most, we identify which state's drift
is the dominant cause of yaw/position error.

Variants:
  normal      - full open-loop rollout (baseline)
  reset_vy    - replace model vy with GT vy at each step
  reset_r     - replace model r with GT r
  reset_delta - replace model delta with mapped-from-steer GT
  reset_vx    - replace model vx with GT vx
  reset_a     - replace model a with mapped-from-throttle GT
  reset_lat   - replace vy + r (all lateral dynamics)
  reset_all5  - replace vx, vy, r, delta, a (only kinematics from model)

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
    p = argparse.ArgumentParser(description="D5: state-reset rollout diagnostic")
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
                   default=ROOT / "results" / "diag_d5_state_reset")
    return p.parse_args()


VARIANTS = [
    "normal",
    "reset_vy",
    "reset_r",
    "reset_delta",
    "reset_vx",
    "reset_a",
    "reset_lat",      # vy + r
    "reset_all5",     # vx, vy, r, delta, a
]

# State indices: X=0, Y=1, psi=2, vx=3, vy=4, r=5, delta=6, a=7


def run_rollout(seg, si, n_steps, dt, p, steer_sign, terrain_fn, variant):
    """Run one rollout with optional state resets."""
    xs = build_state8(
        seg["x"][si], seg["y"][si], seg["yaw"][si],
        seg["vx"][si], seg["vy"][si], seg["r"][si],
        seg["steer"][si], seg["throttle"][si],
        p, steer_sign, "mapped")

    pred = np.zeros((n_steps + 1, 8))
    pred[0] = xs

    for k in range(n_steps):
        j = si + k  # data index

        # Apply state resets from GT BEFORE integration
        state = pred[k].copy()

        if variant == "reset_vy":
            state[4] = float(seg["vy"][j])
        elif variant == "reset_r":
            state[5] = float(seg["r"][j])
        elif variant == "reset_delta":
            state[6] = steer_sign * float(seg["steer"][j]) * p["delta_max"]
        elif variant == "reset_vx":
            state[3] = float(seg["vx"][j])
        elif variant == "reset_a":
            state[7] = np.clip(float(seg["throttle"][j]), 0.0, 1.0) * p["a_max"]
        elif variant == "reset_lat":
            state[4] = float(seg["vy"][j])
            state[5] = float(seg["r"][j])
        elif variant == "reset_all5":
            state[3] = float(seg["vx"][j])
            state[4] = float(seg["vy"][j])
            state[5] = float(seg["r"][j])
            state[6] = steer_sign * float(seg["steer"][j]) * p["delta_max"]
            state[7] = np.clip(float(seg["throttle"][j]), 0.0, 1.0) * p["a_max"]

        u = map_controls(seg["steer"][j], seg["throttle"][j], p, steer_sign)
        pred[k + 1] = rk4_step(state, u, dt, p, terrain_fn)

    return pred


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

    time_axis = np.arange(n_steps + 1) * args.dt

    # Storage: {variant: (n_roll, n_steps+1)} for position and yaw error
    results = {}
    for variant in VARIANTS:
        err_pos = np.zeros((n_roll, n_steps + 1))
        err_yaw = np.zeros((n_roll, n_steps + 1))
        err_vx = np.zeros((n_roll, n_steps + 1))
        err_vy = np.zeros((n_roll, n_steps + 1))
        err_r = np.zeros((n_roll, n_steps + 1))

        for ri, si in enumerate(starts):
            si = int(si)
            gt = np.column_stack([
                seg["x"][si:si + n_steps + 1],
                seg["y"][si:si + n_steps + 1],
                seg["yaw"][si:si + n_steps + 1],
                seg["vx"][si:si + n_steps + 1],
                seg["vy"][si:si + n_steps + 1],
                seg["r"][si:si + n_steps + 1],
            ])

            pred = run_rollout(seg, si, n_steps, args.dt, p, steer_sign,
                               terrain_fn, variant)

            err_pos[ri] = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
            err_yaw[ri] = np.degrees(np.abs(wrap_to_pi(pred[:, 2] - gt[:, 2])))
            err_vx[ri] = pred[:, 3] - gt[:, 3]
            err_vy[ri] = pred[:, 4] - gt[:, 4]
            err_r[ri] = pred[:, 5] - gt[:, 5]

        results[variant] = {
            "pos_median": np.median(err_pos, axis=0),
            "pos_q90": np.percentile(err_pos, 90, axis=0),
            "yaw_median": np.median(err_yaw, axis=0),
            "yaw_q90": np.percentile(err_yaw, 90, axis=0),
            "vx_abs_median": np.median(np.abs(err_vx), axis=0),
            "vy_abs_median": np.median(np.abs(err_vy), axis=0),
            "r_abs_median": np.median(np.abs(np.degrees(err_r)), axis=0),
        }

    # ── Print summary ─────────────────────────────────────────────────────
    print("=" * 80)
    print(f"D5 — STATE-RESET ROLLOUT DIAGNOSTIC ({args.split} split, {n_roll} rollouts)")
    print(f"Config: {args.config.name}")
    print("=" * 80)

    for horizon_s in [0.5, 1.0, 2.0, 3.0, 5.0]:
        idx = min(int(round(horizon_s / args.dt)), n_steps)
        print(f"\n{'─'*80}")
        print(f"At t={horizon_s:.1f}s:")
        print(f"{'Variant':>15s} {'Pos med':>10s} {'Pos Q90':>10s} "
              f"{'Yaw med':>10s} {'Yaw Q90':>10s} {'|vx| med':>10s} "
              f"{'|r| med':>10s}")
        for v in VARIANTS:
            r = results[v]
            print(f"{v:>15s} "
                  f"{r['pos_median'][idx]:>10.3f} "
                  f"{r['pos_q90'][idx]:>10.3f} "
                  f"{r['yaw_median'][idx]:>10.2f} "
                  f"{r['yaw_q90'][idx]:>10.2f} "
                  f"{r['vx_abs_median'][idx]:>10.3f} "
                  f"{r['r_abs_median'][idx]:>10.2f}")

    # Compute reduction ratios at 5s
    print(f"\n{'='*80}")
    print("Error REDUCTION at 5s (lower = more improvement from resetting that state):")
    print(f"{'Variant':>15s} {'Pos ratio':>10s} {'Yaw ratio':>10s}")
    idx5 = n_steps
    normal_pos = results["normal"]["pos_median"][idx5]
    normal_yaw = results["normal"]["yaw_median"][idx5]
    for v in VARIANTS:
        if v == "normal":
            continue
        pos_ratio = results[v]["pos_median"][idx5] / max(normal_pos, 0.01)
        yaw_ratio = results[v]["yaw_median"][idx5] / max(normal_yaw, 0.01)
        print(f"{v:>15s} {pos_ratio:>10.3f} {yaw_ratio:>10.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    colors = {
        "normal": "black",
        "reset_vy": "tab:orange",
        "reset_r": "tab:purple",
        "reset_delta": "tab:red",
        "reset_vx": "tab:green",
        "reset_a": "tab:cyan",
        "reset_lat": "tab:blue",
        "reset_all5": "tab:pink",
    }

    for v in VARIANTS:
        c = colors[v]
        lw = 2.5 if v in ("normal", "reset_all5") else 1.5
        ls = "-" if v not in ("reset_a",) else "--"
        axes[0, 0].plot(time_axis, results[v]["pos_median"], color=c, lw=lw,
                        ls=ls, label=v)
        axes[0, 1].plot(time_axis, results[v]["yaw_median"], color=c, lw=lw,
                        ls=ls, label=v)
        axes[0, 2].plot(time_axis, results[v]["vx_abs_median"], color=c, lw=lw,
                        ls=ls, label=v)
        axes[1, 0].plot(time_axis, results[v]["vy_abs_median"], color=c, lw=lw,
                        ls=ls, label=v)
        axes[1, 1].plot(time_axis, results[v]["r_abs_median"], color=c, lw=lw,
                        ls=ls, label=v)

    for ax, title, ylabel in [
        (axes[0, 0], "Position error (median)", "[m]"),
        (axes[0, 1], "Yaw error (median)", "[deg]"),
        (axes[0, 2], "|vx| error (median)", "[m/s]"),
        (axes[1, 0], "|vy| error (median)", "[m/s]"),
        (axes[1, 1], "|r| error (median)", "[deg/s]"),
    ]:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    # Bar chart: reduction ratio at 5s
    ax = axes[1, 2]
    variants_nobase = [v for v in VARIANTS if v != "normal"]
    pos_ratios = [results[v]["pos_median"][idx5] / max(normal_pos, 0.01)
                  for v in variants_nobase]
    yaw_ratios = [results[v]["yaw_median"][idx5] / max(normal_yaw, 0.01)
                  for v in variants_nobase]
    x_pos = np.arange(len(variants_nobase))
    w = 0.35
    ax.bar(x_pos - w / 2, pos_ratios, w, label="Position", color="tab:red", alpha=0.7)
    ax.bar(x_pos + w / 2, yaw_ratios, w, label="Yaw", color="tab:blue", alpha=0.7)
    ax.axhline(1.0, color="k", ls=":", alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([v.replace("reset_", "") for v in variants_nobase],
                       rotation=45, fontsize=8, ha="right")
    ax.set_ylabel("Error ratio vs normal (lower = more impact)")
    ax.set_title("Impact of resetting each state @ 5s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"D5: State-Reset Rollouts ({args.split}, {n_roll} rollouts)", fontsize=13)
    fig.savefig(args.output_dir / "d5_state_reset.png", dpi=180)
    plt.close(fig)

    # ── Save JSON ─────────────────────────────────────────────────────────
    def to_list(arr):
        return [float(x) for x in arr]

    payload = {
        "config": str(args.config),
        "split": args.split,
        "n_rollouts": n_roll,
        "time_axis": to_list(time_axis),
    }
    for v in VARIANTS:
        payload[v] = {
            "pos_median": to_list(results[v]["pos_median"]),
            "yaw_median": to_list(results[v]["yaw_median"]),
            "pos_q90": to_list(results[v]["pos_q90"]),
            "yaw_q90": to_list(results[v]["yaw_q90"]),
        }
    (args.output_dir / "d5_results.json").write_text(json.dumps(payload, indent=2))

    print(f"\nPlots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
