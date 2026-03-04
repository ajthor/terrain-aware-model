#!/usr/bin/env python3
"""D2 — 1-step residual analysis by regime.

At every timestep, compute model-predicted next state vs actual next state.
Bin residuals by (speed, throttle, |steer|) and report distributions.
Shows WHERE the model is structurally wrong, without rollout compounding.

Uses TRAIN split with tuned (best) config.
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
    p = argparse.ArgumentParser(description="D2: 1-step residual by regime")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs" / "analytical_scintilla_asphalt_best.yaml")
    p.add_argument("--data-root", type=Path,
                   default=ROOT / "function_encoder_beamng")
    p.add_argument("--transmission", default="m1")
    p.add_argument("--terrain", default="asphalt")
    p.add_argument("--split", default="train")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-seconds", type=float, default=250.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "diag_d2_onestep_residuals")
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

    t = np.array(seg["time"], dtype=float)
    n = len(t)
    dt_arr = np.diff(t)
    dt_median = float(np.median(dt_arr[dt_arr > 1e-6]))

    # Filter to only good timesteps (reasonable dt)
    good = (dt_arr > 1e-6) & (dt_arr < 3.0 * dt_median)
    good_idx = np.where(good)[0]
    n_good = len(good_idx)

    print(f"D2: {n_good} valid 1-step transitions out of {n-1} ({args.split} split)")
    print(f"Config: {args.config.name}, dt_median={dt_median:.4f}s")

    # Compute 1-step residuals
    res_vx = np.zeros(n_good)
    res_vy = np.zeros(n_good)
    res_yaw = np.zeros(n_good)
    res_r = np.zeros(n_good)
    res_x = np.zeros(n_good)
    res_y = np.zeros(n_good)

    regime_vx = np.zeros(n_good)
    regime_thr = np.zeros(n_good)
    regime_steer = np.zeros(n_good)
    regime_r = np.zeros(n_good)

    for ii, idx in enumerate(good_idx):
        dt_step = float(dt_arr[idx])
        xs = build_state8(
            seg["x"][idx], seg["y"][idx], seg["yaw"][idx],
            seg["vx"][idx], seg["vy"][idx], seg["r"][idx],
            seg["steer"][idx], seg["throttle"][idx],
            p, steer_sign, "mapped")

        u = map_controls(seg["steer"][idx], seg["throttle"][idx], p, steer_sign)
        xs_next = rk4_step(xs, u, dt_step, p, terrain_fn)

        j = idx + 1
        # Residual = predicted - actual
        res_x[ii] = xs_next[0] - float(seg["x"][j])
        res_y[ii] = xs_next[1] - float(seg["y"][j])
        res_yaw[ii] = float(wrap_to_pi(np.array([xs_next[2] - float(seg["yaw"][j])]))[0])
        res_vx[ii] = xs_next[3] - float(seg["vx"][j])
        res_vy[ii] = xs_next[4] - float(seg["vy"][j])
        res_r[ii] = xs_next[5] - float(seg["r"][j])

        regime_vx[ii] = float(seg["vx"][idx])
        regime_thr[ii] = float(seg["throttle"][idx])
        regime_steer[ii] = float(np.abs(seg["steer"][idx]))
        regime_r[ii] = float(np.abs(seg["r"][idx]))

    # ── Summary statistics ───────────────────────────────────────────────
    print("\n--- Overall 1-step residual stats (pred - GT) ---")
    for name, arr in [("vx", res_vx), ("vy", res_vy), ("yaw[deg]", np.degrees(res_yaw)),
                      ("r[deg/s]", np.degrees(res_r)), ("x", res_x), ("y", res_y)]:
        print(f"  {name:>10s}: mean={np.mean(arr):+.5f}, std={np.std(arr):.5f}, "
              f"|max|={np.max(np.abs(arr)):.5f}")

    # ── Bin by speed ─────────────────────────────────────────────────────
    speed_bins = [0, 5, 10, 15, 20, 100]
    speed_labels = ["0-5", "5-10", "10-15", "15-20", "20+"]

    print("\n--- vx residual by speed bin ---")
    print(f"{'Speed bin':>12s} {'count':>6s} {'mean':>10s} {'std':>10s} {'|mean|/std':>10s}")
    binned_stats = {}
    for i in range(len(speed_bins) - 1):
        mask = (np.abs(regime_vx) >= speed_bins[i]) & (np.abs(regime_vx) < speed_bins[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        m = float(np.mean(res_vx[mask]))
        s = float(np.std(res_vx[mask]))
        ratio = abs(m) / max(s, 1e-9)
        print(f"{speed_labels[i]:>12s} {cnt:>6d} {m:>+10.5f} {s:>10.5f} {ratio:>10.3f}")
        binned_stats[speed_labels[i]] = {"count": cnt, "mean_vx": m, "std_vx": s}

    # ── Bin by throttle ──────────────────────────────────────────────────
    thr_bins = [0, 0.2, 0.5, 0.8, 1.01]
    thr_labels = ["0-0.2", "0.2-0.5", "0.5-0.8", "0.8-1.0"]

    print("\n--- vx residual by throttle bin ---")
    print(f"{'Thr bin':>12s} {'count':>6s} {'mean':>10s} {'std':>10s} {'|mean|/std':>10s}")
    for i in range(len(thr_bins) - 1):
        mask = (regime_thr >= thr_bins[i]) & (regime_thr < thr_bins[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        m = float(np.mean(res_vx[mask]))
        s = float(np.std(res_vx[mask]))
        ratio = abs(m) / max(s, 1e-9)
        print(f"{thr_labels[i]:>12s} {cnt:>6d} {m:>+10.5f} {s:>10.5f} {ratio:>10.3f}")

    # ── Bin by |steer| ───────────────────────────────────────────────────
    steer_bins = [0, 0.05, 0.2, 0.5, 1.01]
    steer_labels = ["<0.05", "0.05-0.2", "0.2-0.5", "0.5-1.0"]

    print("\n--- yaw residual [deg] by |steer| bin ---")
    print(f"{'|Steer| bin':>12s} {'count':>6s} {'mean':>10s} {'std':>10s} {'|mean|/std':>10s}")
    for i in range(len(steer_bins) - 1):
        mask = (regime_steer >= steer_bins[i]) & (regime_steer < steer_bins[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        m = float(np.mean(np.degrees(res_yaw[mask])))
        s = float(np.std(np.degrees(res_yaw[mask])))
        ratio = abs(m) / max(s, 1e-9)
        print(f"{steer_labels[i]:>12s} {cnt:>6d} {m:>+10.5f} {s:>10.5f} {ratio:>10.3f}")

    # ── Combined: high speed + high throttle ─────────────────────────────
    fast_thr_mask = (np.abs(regime_vx) > 12) & (regime_thr > 0.5)
    cnt = int(np.sum(fast_thr_mask))
    print(f"\n--- High speed (>12) + High throttle (>0.5): {cnt} samples ---")
    if cnt > 5:
        print(f"  vx res: mean={np.mean(res_vx[fast_thr_mask]):+.5f}, "
              f"std={np.std(res_vx[fast_thr_mask]):.5f}")
        print(f"  yaw res [deg]: mean={np.mean(np.degrees(res_yaw[fast_thr_mask])):+.5f}, "
              f"std={np.std(np.degrees(res_yaw[fast_thr_mask])):.5f}")

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # Row 1: scatter residuals vs regime
    ax = axes[0, 0]
    ax.scatter(np.abs(regime_vx), res_vx, s=1, alpha=0.15, c="tab:green")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("vx residual [m/s]")
    ax.set_title("vx 1-step residual vs speed")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(regime_thr, res_vx, s=1, alpha=0.15, c="tab:blue")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("Throttle")
    ax.set_ylabel("vx residual [m/s]")
    ax.set_title("vx 1-step residual vs throttle")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(regime_steer, np.degrees(res_yaw), s=1, alpha=0.15, c="tab:red")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|Steer|")
    ax.set_ylabel("Yaw residual [deg]")
    ax.set_title("Yaw 1-step residual vs |steer|")
    ax.grid(alpha=0.3)

    # Row 2: histograms
    ax = axes[1, 0]
    ax.hist(res_vx, bins=100, alpha=0.7, color="tab:green", edgecolor="none")
    ax.axvline(np.mean(res_vx), color="k", ls="--", label=f"mean={np.mean(res_vx):.4f}")
    ax.set_xlabel("vx residual [m/s]")
    ax.set_title("vx residual distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(np.degrees(res_yaw), bins=100, alpha=0.7, color="tab:red", edgecolor="none")
    ax.axvline(np.mean(np.degrees(res_yaw)), color="k", ls="--",
               label=f"mean={np.mean(np.degrees(res_yaw)):.4f}")
    ax.set_xlabel("Yaw residual [deg]")
    ax.set_title("Yaw residual distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.hist(np.degrees(res_r), bins=100, alpha=0.7, color="tab:purple", edgecolor="none")
    ax.axvline(np.mean(np.degrees(res_r)), color="k", ls="--",
               label=f"mean={np.mean(np.degrees(res_r)):.4f}")
    ax.set_xlabel("r residual [deg/s]")
    ax.set_title("Yaw rate residual distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"D2: 1-Step Residuals ({args.split}, {n_good} steps, config={args.config.name})",
                 fontsize=13)
    fig.savefig(args.output_dir / "d2_residuals.png", dpi=180)
    plt.close(fig)

    # ── Binned mean plot ─────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # vx residual mean by speed bin
    ax = axes2[0]
    bin_edges = np.linspace(0, 22, 12)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = []
    bin_stds = []
    for i in range(len(bin_edges) - 1):
        mask = (np.abs(regime_vx) >= bin_edges[i]) & (np.abs(regime_vx) < bin_edges[i + 1])
        if np.sum(mask) >= 5:
            bin_means.append(float(np.mean(res_vx[mask])))
            bin_stds.append(float(np.std(res_vx[mask])))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    ax.bar(bin_centers, bin_means, width=1.8, alpha=0.7, color="tab:green",
           yerr=bin_stds, capsize=3, ecolor="gray")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("Mean vx residual [m/s]")
    ax.set_title("vx bias by speed")
    ax.grid(alpha=0.3)

    # vx residual mean by throttle bin
    ax = axes2[1]
    bin_edges_t = np.linspace(0, 1, 11)
    bin_centers_t = 0.5 * (bin_edges_t[:-1] + bin_edges_t[1:])
    bm, bs = [], []
    for i in range(len(bin_edges_t) - 1):
        mask = (regime_thr >= bin_edges_t[i]) & (regime_thr < bin_edges_t[i + 1])
        if np.sum(mask) >= 5:
            bm.append(float(np.mean(res_vx[mask])))
            bs.append(float(np.std(res_vx[mask])))
        else:
            bm.append(np.nan)
            bs.append(np.nan)
    ax.bar(bin_centers_t, bm, width=0.08, alpha=0.7, color="tab:blue",
           yerr=bs, capsize=3, ecolor="gray")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Mean vx residual [m/s]")
    ax.set_title("vx bias by throttle")
    ax.grid(alpha=0.3)

    # yaw residual mean by speed bin
    ax = axes2[2]
    bm_y, bs_y = [], []
    for i in range(len(bin_edges) - 1):
        mask = (np.abs(regime_vx) >= bin_edges[i]) & (np.abs(regime_vx) < bin_edges[i + 1])
        if np.sum(mask) >= 5:
            bm_y.append(float(np.mean(np.degrees(res_yaw[mask]))))
            bs_y.append(float(np.std(np.degrees(res_yaw[mask]))))
        else:
            bm_y.append(np.nan)
            bs_y.append(np.nan)
    ax.bar(bin_centers, bm_y, width=1.8, alpha=0.7, color="tab:red",
           yerr=bs_y, capsize=3, ecolor="gray")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("Mean yaw residual [deg]")
    ax.set_title("Yaw bias by speed")
    ax.grid(alpha=0.3)

    fig2.suptitle("D2: Binned Mean Residuals", fontsize=13)
    fig2.savefig(args.output_dir / "d2_binned_residuals.png", dpi=180)
    plt.close(fig2)

    # ── Save JSON ────────────────────────────────────────────────────────
    payload = {
        "config": str(args.config),
        "split": args.split,
        "n_valid_steps": n_good,
        "dt_median": dt_median,
        "overall": {
            "vx": {"mean": float(np.mean(res_vx)), "std": float(np.std(res_vx))},
            "vy": {"mean": float(np.mean(res_vy)), "std": float(np.std(res_vy))},
            "yaw_deg": {"mean": float(np.mean(np.degrees(res_yaw))),
                        "std": float(np.std(np.degrees(res_yaw)))},
            "r_degps": {"mean": float(np.mean(np.degrees(res_r))),
                        "std": float(np.std(np.degrees(res_r)))},
        },
    }
    (args.output_dir / "d2_results.json").write_text(json.dumps(payload, indent=2))

    print(f"\nPlots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
