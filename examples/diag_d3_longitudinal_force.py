#!/usr/bin/env python3
"""D3 — Longitudinal force reality check.

Computes implied longitudinal force from data: Fx_implied = m * ax_fd
Compares to model prediction: Fx_model = k_a * a (saturated).
Plots Fx_implied vs throttle, colored by speed.
Shows whether the simple linear k_a model is structurally wrong.

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
    load_model_config,
    load_raw_aligned,
    select_split,
    truncate_duration,
)

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="D3: longitudinal force reality check")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs" / "analytical_scintilla_asphalt_best.yaml")
    p.add_argument("--data-root", type=Path,
                   default=ROOT / "function_encoder_beamng")
    p.add_argument("--transmission", default="m1")
    p.add_argument("--terrain", default="asphalt")
    p.add_argument("--split", default="train")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-seconds", type=float, default=250.0)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "diag_d3_longitudinal_force")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p, terrain_cfg, steer_sign = load_model_config(args.config)

    raw = load_raw_aligned(args.data_root, args.transmission, args.terrain)
    seg = select_split(raw, args.split, args.train_ratio, 1)
    if args.max_seconds > 0:
        seg = truncate_duration(seg, args.max_seconds)

    t = np.array(seg["time"], dtype=float)
    vx = np.array(seg["vx"], dtype=float)
    vy = np.array(seg["vy"], dtype=float)
    yaw_rate = np.array(seg["r"], dtype=float)
    throttle = np.array(seg["throttle"], dtype=float)
    steer = np.array(seg["steer"], dtype=float)

    m = p["m"]
    k_a = p["k_a"]
    a_max = p["a_max"]
    mu = float(terrain_cfg.get("mu_effective", 1.0))
    g = p["g"]

    # Compute data-implied accelerations via finite difference
    ax_fd = np.gradient(vx, t)

    # Centripetal correction: ax_body = dvx/dt - vy*r (in body frame)
    # Actually: dvx/dt = ax_body + vy*r (from kinematics)
    # So: ax_body = dvx/dt - vy*r
    # Wait, the dynamics say: vxdot = r*vy + Fx/m
    # So Fx/m = vxdot - r*vy
    # vxdot = ax_fd (finite diff of vx in body frame)
    ax_body = ax_fd  # vx is already body-frame
    Fx_implied = m * (ax_body - yaw_rate * vy)

    # Model prediction: Fx_model = k_a * a_filtered
    # In steady state with throttle held: a ≈ throttle * a_max (after tau_a settling)
    a_cmd = np.clip(throttle, 0.0, 1.0) * a_max
    # Simple model force (ignoring saturation by mu*Fz for now)
    Fx_model = k_a * a_cmd

    # Also compute with friction saturation
    Fz_tot = m * g  # flat terrain
    Fz_r = 0.5 * Fz_tot  # model uses 50/50 split
    Fx_model_sat = np.clip(Fx_model, -mu * Fz_r, mu * Fz_r)

    # ── Power-limited reference ──────────────────────────────────────────
    # If there's a power limit P, then F_max = P/max(|vx|, eps)
    # Estimate effective power from data
    power_implied = Fx_implied * np.abs(vx)
    vx_abs = np.abs(vx)

    # ── Print summary ────────────────────────────────────────────────────
    print("=" * 70)
    print(f"D3 — LONGITUDINAL FORCE REALITY CHECK ({args.split} split)")
    print(f"Config: {args.config.name}")
    print(f"k_a={k_a}, a_max={a_max}, mu={mu}, m={m}")
    print("=" * 70)

    print(f"\nFx_implied: mean={np.mean(Fx_implied):.1f}N, "
          f"std={np.std(Fx_implied):.1f}N, "
          f"min={np.min(Fx_implied):.1f}N, max={np.max(Fx_implied):.1f}N")
    print(f"Fx_model:   mean={np.mean(Fx_model):.1f}N, "
          f"std={np.std(Fx_model):.1f}N, "
          f"min={np.min(Fx_model):.1f}N, max={np.max(Fx_model):.1f}N")
    print(f"Residual:   mean={np.mean(Fx_implied - Fx_model):.1f}N, "
          f"std={np.std(Fx_implied - Fx_model):.1f}N")

    # Binned analysis
    speed_bins = [(0, 5), (5, 10), (10, 15), (15, 25)]
    thr_bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.01)]

    print("\n--- Fx_implied vs Fx_model by (speed, throttle) ---")
    print(f"{'Speed':>8s} {'Thr':>8s} {'N':>6s} {'Fx_data':>10s} {'Fx_model':>10s} "
          f"{'Residual':>10s} {'Power_kW':>10s}")
    for slo, shi in speed_bins:
        for tlo, thi in thr_bins:
            mask = (vx_abs >= slo) & (vx_abs < shi) & (throttle >= tlo) & (throttle < thi)
            cnt = int(np.sum(mask))
            if cnt < 10:
                continue
            fx_d = float(np.mean(Fx_implied[mask]))
            fx_m = float(np.mean(Fx_model[mask]))
            pwr = float(np.mean(power_implied[mask])) / 1000.0
            print(f"{slo}-{shi:>2d}m/s {tlo:.1f}-{thi:.1f} {cnt:>6d} "
                  f"{fx_d:>+10.0f} {fx_m:>+10.0f} {fx_d - fx_m:>+10.0f} {pwr:>10.1f}")

    # ── Check if power limit exists ──────────────────────────────────────
    # For high throttle (>0.7), look at force vs speed
    high_thr = throttle > 0.7
    if np.sum(high_thr) > 50:
        print("\n--- Force vs speed at high throttle (>0.7) ---")
        for slo, shi in [(0, 5), (5, 8), (8, 11), (11, 14), (14, 17), (17, 22)]:
            mask = high_thr & (vx_abs >= slo) & (vx_abs < shi)
            cnt = int(np.sum(mask))
            if cnt < 5:
                continue
            fx_d = float(np.mean(Fx_implied[mask]))
            pwr = float(np.mean(power_implied[mask])) / 1000.0
            mean_v = float(np.mean(vx_abs[mask]))
            fx_model_val = float(np.mean(Fx_model[mask]))
            print(f"  |vx|={slo:>2d}-{shi:>2d} (mean {mean_v:.1f}): "
                  f"Fx_data={fx_d:>+7.0f}N, Fx_model={fx_model_val:>+7.0f}N, "
                  f"P_data={pwr:>6.1f}kW")

    # ── Plots ────────────────────────────────────────────────────────────

    # Plot 1: Fx_implied vs speed, colored by throttle
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes[0, 0]
    sc = ax.scatter(vx_abs, Fx_implied, c=throttle, s=1, alpha=0.2,
                    cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Throttle")
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("Fx_implied [N]")
    ax.set_title("Data-implied Fx vs speed")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.grid(alpha=0.3)

    # Plot 2: Fx_model vs speed
    ax = axes[0, 1]
    sc = ax.scatter(vx_abs, Fx_model, c=throttle, s=1, alpha=0.2,
                    cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Throttle")
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("Fx_model [N]")
    ax.set_title(f"Model Fx (k_a={k_a}) vs speed")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.grid(alpha=0.3)

    # Plot 3: Residual (data - model) vs speed
    ax = axes[0, 2]
    fx_res = Fx_implied - Fx_model
    sc = ax.scatter(vx_abs, fx_res, c=throttle, s=1, alpha=0.2,
                    cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Throttle")
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("Fx_data - Fx_model [N]")
    ax.set_title("Force residual vs speed")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.grid(alpha=0.3)

    # Plot 4: Implied power vs speed
    ax = axes[1, 0]
    sc = ax.scatter(vx_abs, power_implied / 1000.0, c=throttle, s=1, alpha=0.2,
                    cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Throttle")
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("Implied power [kW]")
    ax.set_title("Data-implied power = Fx * |vx|")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.grid(alpha=0.3)

    # Plot 5: Fx_implied vs Fx_model (direct comparison)
    ax = axes[1, 1]
    ax.scatter(Fx_model, Fx_implied, c=vx_abs, s=1, alpha=0.2,
               cmap="viridis", vmin=0, vmax=20)
    lim = max(np.abs(Fx_model).max(), np.abs(Fx_implied).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x")
    ax.set_xlabel("Fx_model [N]")
    ax.set_ylabel("Fx_data [N]")
    ax.set_title("Data vs Model force (color=speed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 6: Binned mean Fx comparison
    ax = axes[1, 2]
    n_vbins = 15
    vbin_edges = np.linspace(0, 22, n_vbins + 1)
    vbin_centers = 0.5 * (vbin_edges[:-1] + vbin_edges[1:])

    # High throttle only
    ht_mask = throttle > 0.7
    fx_data_binned = []
    fx_model_binned = []
    for i in range(n_vbins):
        mask = ht_mask & (vx_abs >= vbin_edges[i]) & (vx_abs < vbin_edges[i + 1])
        if np.sum(mask) >= 5:
            fx_data_binned.append(float(np.mean(Fx_implied[mask])))
            fx_model_binned.append(float(np.mean(Fx_model[mask])))
        else:
            fx_data_binned.append(np.nan)
            fx_model_binned.append(np.nan)

    w = 0.6
    ax.bar(vbin_centers - w / 2, fx_data_binned, width=w, alpha=0.7,
           color="tab:blue", label="Fx data")
    ax.bar(vbin_centers + w / 2, fx_model_binned, width=w, alpha=0.7,
           color="tab:orange", label="Fx model")
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("Mean Fx [N]")
    ax.set_title("Mean Fx at high throttle (>0.7)")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"D3: Longitudinal Force ({args.split}, config={args.config.name})",
                 fontsize=13)
    fig.savefig(args.output_dir / "d3_longitudinal_force.png", dpi=180)
    plt.close(fig)

    # ── Engine braking / coast analysis ──────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Low throttle regime: coast / engine braking
    coast_mask = throttle < 0.1
    if np.sum(coast_mask) > 50:
        ax1.scatter(vx_abs[coast_mask], Fx_implied[coast_mask],
                    s=2, alpha=0.3, c="tab:red", label="Fx data (thr<0.1)")
        # Fit a linear coast model: Fx = -c0 - c1*|vx|
        vx_coast = vx_abs[coast_mask]
        fx_coast = Fx_implied[coast_mask]
        valid = vx_coast > 1.0  # avoid near-zero weirdness
        if np.sum(valid) > 20:
            A = np.column_stack([np.ones(np.sum(valid)), vx_coast[valid]])
            coeffs = np.linalg.lstsq(A, fx_coast[valid], rcond=None)[0]
            vx_fit = np.linspace(1, 22, 50)
            fx_fit = coeffs[0] + coeffs[1] * vx_fit
            ax1.plot(vx_fit, fx_fit, "k--", lw=2,
                     label=f"Fit: {coeffs[0]:.0f} + {coeffs[1]:.0f}*|vx|")
            print(f"\n--- Coast/engine braking fit (throttle<0.1) ---")
            print(f"  Fx ≈ {coeffs[0]:.1f} + {coeffs[1]:.1f} * |vx|")
            print(f"  (Model has NO coast force → always Fx≈0 at zero throttle)")

        ax1.axhline(0, color="gray", ls=":", alpha=0.3)
        ax1.set_xlabel("|vx| [m/s]")
        ax1.set_ylabel("Fx [N]")
        ax1.set_title("Coast / engine braking (throttle < 0.1)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

    # Time-domain comparison
    seg_start = len(t) // 3
    seg_len = min(1000, len(t) - seg_start)
    sl = slice(seg_start, seg_start + seg_len)
    t_seg = t[sl] - t[seg_start]

    ax2.plot(t_seg, Fx_implied[sl], "b-", lw=1, alpha=0.7, label="Fx data")
    ax2.plot(t_seg, Fx_model[sl], "r-", lw=1, alpha=0.7, label="Fx model")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Fx [N]")
    ax2.set_title("Force time series (20s segment)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig2.suptitle("D3: Coast Forces & Time Comparison", fontsize=13)
    fig2.savefig(args.output_dir / "d3_coast_and_timeseries.png", dpi=180)
    plt.close(fig2)

    # ── Save JSON ────────────────────────────────────────────────────────
    payload = {
        "config": str(args.config),
        "split": args.split,
        "model_params": {"k_a": k_a, "a_max": a_max, "mu": mu, "m": m},
        "fx_implied": {
            "mean": float(np.mean(Fx_implied)),
            "std": float(np.std(Fx_implied)),
        },
        "fx_model": {
            "mean": float(np.mean(Fx_model)),
            "std": float(np.std(Fx_model)),
        },
        "residual_mean": float(np.mean(Fx_implied - Fx_model)),
        "residual_std": float(np.std(Fx_implied - Fx_model)),
    }
    (args.output_dir / "d3_results.json").write_text(json.dumps(payload, indent=2))

    print(f"\nPlots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
