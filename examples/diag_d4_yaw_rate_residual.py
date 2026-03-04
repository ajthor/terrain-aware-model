#!/usr/bin/env python3
"""D4 — Yaw rate residual decomposition.

Computes 1-step yaw rate (r) residuals and bins them by:
  - |steer| (steering magnitude) — to detect tire saturation
  - |vx| (speed) — to detect speed-dependent understeer
  - |alpha_f| (front slip angle) — to see if linear tire model breaks down
  - transient vs steady-state steering — to detect actuator lag issues

Also computes model-predicted vs data-implied lateral forces to see
where the linear tire model diverges from reality.

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
from terrain_aware_model import vehicle_dynamics

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="D4: yaw rate residual decomposition")
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
                   default=ROOT / "results" / "diag_d4_yaw_rate")
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

    good = (dt_arr > 1e-6) & (dt_arr < 3.0 * dt_median)
    good_idx = np.where(good)[0]
    n_good = len(good_idx)

    print(f"D4: {n_good} valid transitions ({args.split} split)")
    print(f"Config: {args.config.name}, dt_median={dt_median:.4f}s")

    # Storage
    res_r = np.zeros(n_good)           # yaw rate residual (pred - GT) [rad/s]
    res_yaw = np.zeros(n_good)         # yaw residual [rad]
    regime_vx = np.zeros(n_good)
    regime_steer = np.zeros(n_good)    # signed steer
    regime_abs_steer = np.zeros(n_good)
    regime_r_gt = np.zeros(n_good)     # GT yaw rate
    regime_alpha_f = np.zeros(n_good)  # front slip angle
    regime_alpha_r = np.zeros(n_good)  # rear slip angle
    regime_delta = np.zeros(n_good)    # model steering angle
    regime_steer_rate = np.zeros(n_good)  # d(steer)/dt
    model_Fy_f = np.zeros(n_good)      # model front lateral force
    model_Fy_r = np.zeros(n_good)      # model rear lateral force
    data_rdot = np.zeros(n_good)       # data-implied rdot
    model_rdot = np.zeros(n_good)      # model-predicted rdot

    lf = float(p["lf"])
    lr = float(p["lr"])
    Iz = float(p["Iz"])
    Cf0 = float(p["Cf0"])
    Cr0 = float(p["Cr0"])
    mass = float(p["m"])
    k_lat_f = float(p.get("k_lat_f", 0.0))
    k_lat_r = float(p.get("k_lat_r", 0.0))
    g = float(p["g"])
    mu_eff = float(terrain_cfg.get("mu_effective", 1.0))
    Fz_f_approx = 0.5 * mass * g  # rough normal force on front axle

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
        res_r[ii] = xs_next[5] - float(seg["r"][j])
        res_yaw[ii] = float(wrap_to_pi(np.array([xs_next[2] - float(seg["yaw"][j])]))[0])

        vx = float(seg["vx"][idx])
        vy = float(seg["vy"][idx])
        r_gt = float(seg["r"][idx])
        delta = float(xs[6])  # model delta state

        regime_vx[ii] = vx
        regime_steer[ii] = float(seg["steer"][idx])
        regime_abs_steer[ii] = abs(float(seg["steer"][idx]))
        regime_r_gt[ii] = r_gt
        regime_delta[ii] = delta

        # Compute slip angles
        vx_eff = max(abs(vx), 0.2) * (1.0 if vx >= 0 else -1.0)
        alpha_f = delta - np.arctan2(vy + lf * r_gt, vx_eff)
        alpha_r = -np.arctan2(vy - lr * r_gt, vx_eff)
        regime_alpha_f[ii] = alpha_f
        regime_alpha_r[ii] = alpha_r

        # Model lateral forces (linear)
        Cf = Cf0  # ignoring ay correction for this diagnostic
        Cr = Cr0
        Fy_f_lin = Cf * alpha_f
        Fy_r_lin = Cr * alpha_r
        model_Fy_f[ii] = Fy_f_lin
        model_Fy_r[ii] = Fy_r_lin

        # Steer rate
        if idx > 0 and idx < n - 1:
            dt_prev = float(t[idx] - t[idx - 1])
            if dt_prev > 1e-6:
                regime_steer_rate[ii] = (float(seg["steer"][idx]) - float(seg["steer"][idx - 1])) / dt_prev

        # Data-implied rdot
        r_next_gt = float(seg["r"][j])
        data_rdot[ii] = (r_next_gt - r_gt) / dt_step

        # Model rdot from dynamics
        xdot = vehicle_dynamics(0.0, xs, u, p, terrain_fn)
        model_rdot[ii] = float(xdot[5])

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("D4 — YAW RATE RESIDUAL DECOMPOSITION")
    print("=" * 70)

    print(f"\nOverall r residual: mean={np.mean(np.degrees(res_r)):+.4f} deg/s, "
          f"std={np.std(np.degrees(res_r)):.4f} deg/s")
    print(f"Overall rdot residual: mean={np.mean(np.degrees(model_rdot - data_rdot)):+.4f} deg/s², "
          f"std={np.std(np.degrees(model_rdot - data_rdot)):.4f} deg/s²")

    # ── Bin r residual by |steer| ─────────────────────────────────────────
    steer_edges = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
    steer_labels = ["<0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", "0.40+"]

    print("\n--- r residual [deg/s] by |steer| ---")
    print(f"{'|Steer|':>12s} {'count':>6s} {'mean':>10s} {'std':>10s} {'|mean|/std':>10s}")
    r_by_steer = {}
    for i in range(len(steer_edges) - 1):
        mask = (regime_abs_steer >= steer_edges[i]) & (regime_abs_steer < steer_edges[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        vals = np.degrees(res_r[mask])
        bm, bs = float(np.mean(vals)), float(np.std(vals))
        ratio = abs(bm) / max(bs, 1e-9)
        print(f"{steer_labels[i]:>12s} {cnt:>6d} {bm:>+10.4f} {bs:>10.4f} {ratio:>10.3f}")
        r_by_steer[steer_labels[i]] = {"count": cnt, "mean": bm, "std": bs}

    # ── Bin r residual by |vx| ────────────────────────────────────────────
    speed_edges = [0, 3, 6, 10, 15, 20, 100]
    speed_labels = ["0-3", "3-6", "6-10", "10-15", "15-20", "20+"]

    print("\n--- r residual [deg/s] by speed ---")
    print(f"{'Speed':>12s} {'count':>6s} {'mean':>10s} {'std':>10s} {'|mean|/std':>10s}")
    r_by_speed = {}
    for i in range(len(speed_edges) - 1):
        mask = (np.abs(regime_vx) >= speed_edges[i]) & (np.abs(regime_vx) < speed_edges[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        vals = np.degrees(res_r[mask])
        bm, bs = float(np.mean(vals)), float(np.std(vals))
        ratio = abs(bm) / max(bs, 1e-9)
        print(f"{speed_labels[i]:>12s} {cnt:>6d} {bm:>+10.4f} {bs:>10.4f} {ratio:>10.3f}")
        r_by_speed[speed_labels[i]] = {"count": cnt, "mean": bm, "std": bs}

    # ── Bin r residual by |alpha_f| (front slip angle) ────────────────────
    alpha_edges_deg = [0, 1, 2, 4, 7, 12, 90]
    alpha_labels = ["<1°", "1-2°", "2-4°", "4-7°", "7-12°", "12+°"]

    print("\n--- r residual [deg/s] by |alpha_f| (front slip angle) ---")
    print(f"{'|alpha_f|':>12s} {'count':>6s} {'r mean':>10s} {'r std':>10s} "
          f"{'Fy_f model':>12s} {'Fy_f_sat':>12s}")
    r_by_alpha = {}
    for i in range(len(alpha_edges_deg) - 1):
        lo_rad = np.radians(alpha_edges_deg[i])
        hi_rad = np.radians(alpha_edges_deg[i + 1])
        mask = (np.abs(regime_alpha_f) >= lo_rad) & (np.abs(regime_alpha_f) < hi_rad)
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        vals_r = np.degrees(res_r[mask])
        Fy_f_mean = float(np.mean(np.abs(model_Fy_f[mask])))
        Fy_f_sat_limit = mu_eff * Fz_f_approx
        m_r, s_r = float(np.mean(vals_r)), float(np.std(vals_r))
        print(f"{alpha_labels[i]:>12s} {cnt:>6d} {m_r:>+10.4f} {s_r:>10.4f} "
              f"{Fy_f_mean:>12.0f} {Fy_f_sat_limit:>12.0f}")
        r_by_alpha[alpha_labels[i]] = {
            "count": cnt, "r_mean": m_r, "r_std": s_r,
            "Fy_f_model_mean": Fy_f_mean,
            "Fy_f_sat_limit": float(Fy_f_sat_limit),
            "ratio_to_sat": Fy_f_mean / max(Fy_f_sat_limit, 1.0),
        }

    # ── Transient vs steady-state steering ────────────────────────────────
    abs_steer_rate = np.abs(regime_steer_rate)
    steer_rate_med = float(np.median(abs_steer_rate[abs_steer_rate > 0.01]))

    print(f"\n--- Transient vs steady-state (|d_steer/dt| threshold={steer_rate_med:.3f}) ---")
    steady_mask = abs_steer_rate < steer_rate_med
    trans_mask = abs_steer_rate >= steer_rate_med

    for label, mask in [("Steady", steady_mask), ("Transient", trans_mask)]:
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        vals = np.degrees(res_r[mask])
        m_v, s_v = float(np.mean(vals)), float(np.std(vals))
        print(f"  {label:>10s}: n={cnt:>5d}, r_res mean={m_v:+.4f} deg/s, std={s_v:.4f} deg/s")

    # ── Bin rdot residual by |steer| (to see moment error) ────────────────
    rdot_res = np.degrees(model_rdot - data_rdot)

    print("\n--- rdot residual [deg/s²] by |steer| ---")
    print(f"{'|Steer|':>12s} {'count':>6s} {'mean':>10s} {'std':>10s}")
    for i in range(len(steer_edges) - 1):
        mask = (regime_abs_steer >= steer_edges[i]) & (regime_abs_steer < steer_edges[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        vals = rdot_res[mask]
        print(f"{steer_labels[i]:>12s} {cnt:>6d} {np.mean(vals):>+10.4f} {np.std(vals):>10.4f}")

    # ── Combined: high steer + high speed ────────────────────────────────
    high_steer_fast = (regime_abs_steer > 0.1) & (np.abs(regime_vx) > 10)
    cnt = int(np.sum(high_steer_fast))
    print(f"\n--- High steer (>0.1) + High speed (>10 m/s): {cnt} samples ---")
    if cnt >= 5:
        vals = np.degrees(res_r[high_steer_fast])
        print(f"  r residual: mean={np.mean(vals):+.4f}, std={np.std(vals):.4f} deg/s")
        vals_rdot = rdot_res[high_steer_fast]
        print(f"  rdot residual: mean={np.mean(vals_rdot):+.4f}, std={np.std(vals_rdot):.4f} deg/s²")

    # ── PLOTS ─────────────────────────────────────────────────────────────

    # Plot 1: r residual decomposition (4 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(regime_abs_steer, np.degrees(res_r), s=1, alpha=0.15, c="tab:purple")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|Steer input|")
    ax.set_ylabel("r residual [deg/s]")
    ax.set_title("r residual vs |steer|")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(np.abs(regime_vx), np.degrees(res_r), s=1, alpha=0.15, c="tab:blue")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("r residual [deg/s]")
    ax.set_title("r residual vs speed")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(np.degrees(np.abs(regime_alpha_f)), np.degrees(res_r), s=1, alpha=0.15, c="tab:red")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|alpha_f| [deg]")
    ax.set_ylabel("r residual [deg/s]")
    ax.set_title("r residual vs front slip angle")
    ax.grid(alpha=0.3)

    # Model Fy_f vs alpha_f (to see saturation)
    ax = axes[1, 0]
    sort_idx = np.argsort(np.degrees(regime_alpha_f))
    ax.scatter(np.degrees(regime_alpha_f), model_Fy_f / 1000, s=1, alpha=0.15, c="tab:green")
    # Overlay linear line
    alpha_range = np.linspace(-15, 15, 100)
    ax.plot(alpha_range, Cf0 * np.radians(alpha_range) / 1000, "r--", lw=2,
            label=f"Cf0={Cf0:.0f} (linear)")
    # Overlay saturation limit
    ax.axhline(mu_eff * Fz_f_approx / 1000, color="orange", ls=":", lw=2,
               label=f"mu*Fz_f={mu_eff * Fz_f_approx:.0f}N")
    ax.axhline(-mu_eff * Fz_f_approx / 1000, color="orange", ls=":", lw=2)
    ax.set_xlabel("alpha_f [deg]")
    ax.set_ylabel("Fy_f [kN]")
    ax.set_title("Model front lateral force vs slip angle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-20, 20)

    # rdot: model vs data
    ax = axes[1, 1]
    rdot_range = max(np.percentile(np.abs(np.degrees(data_rdot)), 95),
                     np.percentile(np.abs(np.degrees(model_rdot)), 95))
    ax.scatter(np.degrees(data_rdot), np.degrees(model_rdot), s=1, alpha=0.1, c="tab:purple")
    ax.plot([-rdot_range, rdot_range], [-rdot_range, rdot_range], "r--", lw=1.5, label="y=x")
    ax.set_xlabel("Data rdot [deg/s²]")
    ax.set_ylabel("Model rdot [deg/s²]")
    ax.set_title("Model vs data yaw acceleration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    lim = rdot_range * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Binned r residual std by |steer| and speed (heatmap)
    ax = axes[1, 2]
    steer_bin_edges = np.array([0, 0.05, 0.1, 0.2, 0.4, 1.01])
    speed_bin_edges = np.array([0, 5, 10, 15, 20, 100])
    heatmap = np.full((len(speed_bin_edges) - 1, len(steer_bin_edges) - 1), np.nan)
    for si in range(len(steer_bin_edges) - 1):
        for vi in range(len(speed_bin_edges) - 1):
            mask = ((regime_abs_steer >= steer_bin_edges[si]) &
                    (regime_abs_steer < steer_bin_edges[si + 1]) &
                    (np.abs(regime_vx) >= speed_bin_edges[vi]) &
                    (np.abs(regime_vx) < speed_bin_edges[vi + 1]))
            if np.sum(mask) >= 10:
                heatmap[vi, si] = float(np.std(np.degrees(res_r[mask])))

    im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="YlOrRd",
                   extent=[0, len(steer_bin_edges) - 1, 0, len(speed_bin_edges) - 1])
    ax.set_xticks(np.arange(len(steer_bin_edges) - 1) + 0.5)
    ax.set_xticklabels(["<.05", ".05-.1", ".1-.2", ".2-.4", ".4+"], fontsize=8)
    ax.set_yticks(np.arange(len(speed_bin_edges) - 1) + 0.5)
    ax.set_yticklabels(["0-5", "5-10", "10-15", "15-20", "20+"], fontsize=8)
    ax.set_xlabel("|Steer|")
    ax.set_ylabel("|vx| [m/s]")
    ax.set_title("r residual STD [deg/s] by regime")
    # Annotate cells
    for si in range(heatmap.shape[1]):
        for vi in range(heatmap.shape[0]):
            val = heatmap[vi, si]
            if not np.isnan(val):
                ax.text(si + 0.5, vi + 0.5, f"{val:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"D4: Yaw Rate Residual Decomposition ({args.split}, {n_good} steps)", fontsize=13)
    fig.savefig(args.output_dir / "d4_yaw_rate_decomposition.png", dpi=180)
    plt.close(fig)

    # Plot 2: Fy_f saturation check — linear force vs saturation limit by alpha bin
    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    alpha_f_deg = np.degrees(np.abs(regime_alpha_f))
    alpha_bin_edges = np.array([0, 1, 2, 3, 5, 8, 12, 20, 90])
    alpha_bin_labels = [f"{alpha_bin_edges[i]}-{alpha_bin_edges[i+1]}"
                        for i in range(len(alpha_bin_edges) - 1)]
    bin_centers = []
    bin_mean_Fy = []
    bin_sat_ratio = []
    bin_r_std = []
    for i in range(len(alpha_bin_edges) - 1):
        mask = (alpha_f_deg >= alpha_bin_edges[i]) & (alpha_f_deg < alpha_bin_edges[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        center = 0.5 * (alpha_bin_edges[i] + alpha_bin_edges[i + 1])
        bin_centers.append(center)
        Fy_abs = float(np.mean(np.abs(model_Fy_f[mask])))
        bin_mean_Fy.append(Fy_abs)
        bin_sat_ratio.append(Fy_abs / max(mu_eff * Fz_f_approx, 1.0))
        bin_r_std.append(float(np.std(np.degrees(res_r[mask]))))

    ax_a.bar(range(len(bin_centers)), bin_sat_ratio, color="tab:orange", alpha=0.7)
    ax_a.axhline(1.0, color="red", ls="--", lw=2, label="Saturation limit (mu*Fz)")
    ax_a.set_xticks(range(len(bin_centers)))
    ax_a.set_xticklabels([f"{c:.0f}°" for c in bin_centers], fontsize=8)
    ax_a.set_xlabel("|alpha_f| bin center")
    ax_a.set_ylabel("Model |Fy_f| / (mu*Fz_f)")
    ax_a.set_title("Front tire force ratio to saturation limit")
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    ax_b.bar(range(len(bin_centers)), bin_r_std, color="tab:purple", alpha=0.7)
    ax_b.set_xticks(range(len(bin_centers)))
    ax_b.set_xticklabels([f"{c:.0f}°" for c in bin_centers], fontsize=8)
    ax_b.set_xlabel("|alpha_f| bin center")
    ax_b.set_ylabel("r residual STD [deg/s]")
    ax_b.set_title("Yaw rate residual variance by front slip angle")
    ax_b.grid(alpha=0.3)

    fig2.suptitle("D4: Tire Saturation Check", fontsize=13)
    fig2.savefig(args.output_dir / "d4_tire_saturation.png", dpi=180)
    plt.close(fig2)

    # ── Save JSON ─────────────────────────────────────────────────────────
    payload = {
        "config": str(args.config),
        "split": args.split,
        "n_valid_steps": n_good,
        "overall_r_residual_degps": {
            "mean": float(np.mean(np.degrees(res_r))),
            "std": float(np.std(np.degrees(res_r))),
        },
        "overall_rdot_residual_degps2": {
            "mean": float(np.mean(rdot_res)),
            "std": float(np.std(rdot_res)),
        },
        "r_by_steer_bin": r_by_steer,
        "r_by_speed_bin": r_by_speed,
        "r_by_alpha_f_bin": r_by_alpha,
        "model_params": {
            "Cf0": Cf0, "Cr0": Cr0, "lf": lf, "lr": lr, "Iz": Iz,
            "mu_effective": mu_eff,
        },
    }
    (args.output_dir / "d4_results.json").write_text(json.dumps(payload, indent=2))

    print(f"\nPlots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
