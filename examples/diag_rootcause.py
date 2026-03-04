#!/usr/bin/env python3
"""Root cause analysis: What exactly drives the remaining yaw error?

1. 1-step rdot residual decomposition: how much comes from the polynomial
   residual vs the analytical model?
2. δ̇ → Mz correction: is the relationship truly linear, or is there
   saturation / nonlinearity that res_w3 can't capture?
3. Steering transient analysis: does the error concentrate at steering
   sign-change events?
4. Residual with res_w3=0 vs with res_w3: how much does res_w3 actually fix?
5. What does the remaining error after res_w3 look like?
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
    p = argparse.ArgumentParser(description="Root cause analysis")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs" / "analytical_scintilla_asphalt_pacejka_best.yaml")
    p.add_argument("--data-root", type=Path,
                   default=ROOT / "function_encoder_beamng")
    p.add_argument("--transmission", default="m1")
    p.add_argument("--terrain", default="asphalt")
    p.add_argument("--split", default="train")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-seconds", type=float, default=250.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "diag_rootcause")
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

    print(f"Root cause analysis: {n_good} valid transitions ({args.split} split)")
    print(f"Config: {args.config.name}")

    # Extract key params
    tau_delta = max(float(p["tau_delta"]), 1e-3)
    delta_max = float(p["delta_max"])
    res_w3 = float(p.get("res_w3", 0.0))
    res_w4 = float(p.get("res_w4", 0.0))
    Iz = float(p["Iz"])

    # Storage
    data_rdot = np.zeros(n_good)          # GT rdot from finite diff
    model_rdot_full = np.zeros(n_good)    # model rdot with all residual
    model_rdot_nores = np.zeros(n_good)   # model rdot WITHOUT polynomial residual
    model_rdot_no_w3w4 = np.zeros(n_good) # model rdot without w3,w4 only

    regime_vx = np.zeros(n_good)
    regime_steer = np.zeros(n_good)
    regime_delta_rate = np.zeros(n_good)   # model δ̇ = (δ_cmd - δ) / τ_δ
    regime_steer_rate = np.zeros(n_good)   # raw steer input rate d(steer)/dt
    regime_r = np.zeros(n_good)
    regime_vy = np.zeros(n_good)

    # Also compute what the polynomial residual contributes
    poly_Mz = np.zeros(n_good)           # full polynomial correction
    poly_w3_contrib = np.zeros(n_good)    # just w3*delta_rate
    poly_w4_contrib = np.zeros(n_good)    # just w4*delta_rate*vx

    for ii, idx in enumerate(good_idx):
        dt_step = float(dt_arr[idx])
        xs = build_state8(
            seg["x"][idx], seg["y"][idx], seg["yaw"][idx],
            seg["vx"][idx], seg["vy"][idx], seg["r"][idx],
            seg["steer"][idx], seg["throttle"][idx],
            p, steer_sign, "mapped")

        u = map_controls(seg["steer"][idx], seg["throttle"][idx], p, steer_sign)

        # Full model rdot
        xdot_full = vehicle_dynamics(0.0, xs, u, p, terrain_fn)
        model_rdot_full[ii] = float(xdot_full[5])

        # Model rdot WITHOUT polynomial residual (set all res_w to 0)
        p_nores = dict(p)
        for w in ("res_w0", "res_w1", "res_w2", "res_w3", "res_w4", "res_w5"):
            p_nores[w] = 0.0
        xdot_nores = vehicle_dynamics(0.0, xs, u, p_nores, terrain_fn)
        model_rdot_nores[ii] = float(xdot_nores[5])

        # Model rdot without w3, w4 only
        p_no_w3w4 = dict(p)
        p_no_w3w4["res_w3"] = 0.0
        p_no_w3w4["res_w4"] = 0.0
        xdot_no_w3w4 = vehicle_dynamics(0.0, xs, u, p_no_w3w4, terrain_fn)
        model_rdot_no_w3w4[ii] = float(xdot_no_w3w4[5])

        # GT rdot
        j = idx + 1
        data_rdot[ii] = (float(seg["r"][j]) - float(seg["r"][idx])) / dt_step

        # Regime features
        vx_val = float(seg["vx"][idx])
        regime_vx[ii] = vx_val
        regime_steer[ii] = float(seg["steer"][idx])
        regime_r[ii] = float(seg["r"][idx])
        regime_vy[ii] = float(seg["vy"][idx])

        # Model delta rate
        delta = float(xs[6])
        delta_cmd = float(u[0])
        delta_rate = (delta_cmd - delta) / tau_delta
        regime_delta_rate[ii] = delta_rate

        # Raw steer rate
        if idx > 0:
            dt_prev = float(t[idx] - t[idx - 1])
            if dt_prev > 1e-6:
                regime_steer_rate[ii] = (float(seg["steer"][idx]) - float(seg["steer"][idx - 1])) / dt_prev

        # Polynomial contributions
        vx_eff = max(abs(vx_val), 0.2) * (1.0 if vx_val >= 0 else -1.0)
        r_val = float(seg["r"][idx])
        vy_val = float(seg["vy"][idx])
        w0 = float(p.get("res_w0", 0.0))
        w1 = float(p.get("res_w1", 0.0))
        w2 = float(p.get("res_w2", 0.0))
        w3 = float(p.get("res_w3", 0.0))
        w4 = float(p.get("res_w4", 0.0))
        w5 = float(p.get("res_w5", 0.0))
        poly_full = w0 + w1*r_val + w2*r_val*vx_eff + w3*delta_rate + w4*delta_rate*vx_eff + w5*vy_val
        poly_Mz[ii] = poly_full
        poly_w3_contrib[ii] = w3 * delta_rate
        poly_w4_contrib[ii] = w4 * delta_rate * vx_eff

    # ── Compute residuals ─────────────────────────────────────────────────
    rdot_res_full = model_rdot_full - data_rdot         # with all corrections
    rdot_res_nores = model_rdot_nores - data_rdot       # without polynomial
    rdot_res_no_w3w4 = model_rdot_no_w3w4 - data_rdot  # without w3,w4

    # Convert to degrees/s²
    rdot_res_full_d = np.degrees(rdot_res_full)
    rdot_res_nores_d = np.degrees(rdot_res_nores)
    rdot_res_no_w3w4_d = np.degrees(rdot_res_no_w3w4)

    # ── Analysis 1: How much does the polynomial residual help? ──────────
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Polynomial residual contribution")
    print("=" * 80)
    print(f"  rdot residual WITHOUT polynomial: mean={np.mean(rdot_res_nores_d):+.4f}, "
          f"std={np.std(rdot_res_nores_d):.4f} deg/s²")
    print(f"  rdot residual WITH full poly:     mean={np.mean(rdot_res_full_d):+.4f}, "
          f"std={np.std(rdot_res_full_d):.4f} deg/s²")
    print(f"  rdot residual WITHOUT w3,w4 only: mean={np.mean(rdot_res_no_w3w4_d):+.4f}, "
          f"std={np.std(rdot_res_no_w3w4_d):.4f} deg/s²")
    print(f"\n  STD reduction from full poly: {np.std(rdot_res_nores_d):.4f} -> {np.std(rdot_res_full_d):.4f} "
          f"({100*(1 - np.std(rdot_res_full_d)/np.std(rdot_res_nores_d)):.1f}%)")
    print(f"  STD reduction from w3+w4 alone: {np.std(rdot_res_no_w3w4_d):.4f} -> {np.std(rdot_res_full_d):.4f} "
          f"({100*(1 - np.std(rdot_res_full_d)/np.std(rdot_res_no_w3w4_d)):.1f}%)")

    # ── Analysis 2: δ̇ nonlinearity ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYSIS 2: δ̇ → rdot error nonlinearity")
    print("=" * 80)

    # The "error that w3 is trying to fix" = rdot_res_no_w3w4
    # If the relationship is linear, res_w3 * delta_rate should capture most of it.
    # Plot the residual without w3w4 vs delta_rate to see if it's linear.
    abs_dr = np.abs(regime_delta_rate)
    dr_bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 100]
    dr_labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", "0.5-1", "1-2", "2-5", "5+"]

    print(f"\n{'|δ̇| bin':>12s} {'count':>6s} {'res mean':>10s} {'res std':>10s} "
          f"{'w3 fix':>10s} {'remaining':>10s}")
    for i in range(len(dr_bins) - 1):
        mask = (abs_dr >= dr_bins[i]) & (abs_dr < dr_bins[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 10:
            continue
        res_mean = float(np.mean(rdot_res_no_w3w4_d[mask]))
        res_std = float(np.std(rdot_res_no_w3w4_d[mask]))
        w3_fix_mean = float(np.mean(np.degrees(poly_w3_contrib[mask] + poly_w4_contrib[mask]) / Iz))
        remaining = float(np.mean(rdot_res_full_d[mask]))
        print(f"{dr_labels[i]:>12s} {cnt:>6d} {res_mean:>+10.4f} {res_std:>10.4f} "
              f"{w3_fix_mean:>+10.4f} {remaining:>+10.4f}")

    # ── Analysis 3: Steering transient events ─────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Error at steering transient events")
    print("=" * 80)

    # Define "transient" as |steer_rate| > threshold
    abs_steer_rate = np.abs(regime_steer_rate)
    sr_thresh = float(np.percentile(abs_steer_rate[abs_steer_rate > 0.01], 75))
    transient = abs_steer_rate > sr_thresh
    steady = abs_steer_rate <= sr_thresh

    print(f"  Steer rate threshold (P75): {sr_thresh:.3f} /s")
    print(f"  Steady samples: {int(np.sum(steady))}, Transient samples: {int(np.sum(transient))}")

    for label, mask in [("Steady", steady), ("Transient", transient)]:
        cnt = int(np.sum(mask))
        if cnt < 10:
            continue
        res_std_full = float(np.std(rdot_res_full_d[mask]))
        res_std_nores = float(np.std(rdot_res_nores_d[mask]))
        res_mean_full = float(np.mean(rdot_res_full_d[mask]))
        print(f"  {label:>10s}: n={cnt:5d}, rdot res (full): mean={res_mean_full:+.4f}, std={res_std_full:.4f} deg/s²"
              f"  (without poly: std={res_std_nores:.4f})")

    # ── Analysis 4: Sign of δ̇ matters? ────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Asymmetry in δ̇ correction")
    print("=" * 80)

    dr_pos = regime_delta_rate > 0.1  # steering left
    dr_neg = regime_delta_rate < -0.1  # steering right
    dr_zero = np.abs(regime_delta_rate) <= 0.1  # roughly straight

    for label, mask in [("δ̇ > 0.1 (left)", dr_pos), ("δ̇ < -0.1 (right)", dr_neg),
                        ("|δ̇| ≤ 0.1", dr_zero)]:
        cnt = int(np.sum(mask))
        if cnt < 10:
            continue
        res_nopoly = float(np.mean(rdot_res_nores_d[mask]))
        res_full = float(np.mean(rdot_res_full_d[mask]))
        res_std = float(np.std(rdot_res_full_d[mask]))
        print(f"  {label:>20s}: n={cnt:5d}, "
              f"res w/o poly={res_nopoly:+.4f}, "
              f"res w/ poly={res_full:+.4f}, std={res_std:.4f} deg/s²")

    # ── Analysis 5: Speed-dependent structure of remaining error ──────────
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Remaining rdot error (WITH full polynomial) by speed")
    print("=" * 80)

    speed_bins = [0, 3, 6, 10, 15, 20, 100]
    speed_labels = ["0-3", "3-6", "6-10", "10-15", "15-20", "20+"]

    print(f"{'Speed':>10s} {'count':>6s} {'mean':>10s} {'std':>10s} {'Q5':>10s} {'Q95':>10s}")
    for i in range(len(speed_bins) - 1):
        mask = (np.abs(regime_vx) >= speed_bins[i]) & (np.abs(regime_vx) < speed_bins[i + 1])
        cnt = int(np.sum(mask))
        if cnt < 10:
            continue
        vals = rdot_res_full_d[mask]
        print(f"{speed_labels[i]:>10s} {cnt:>6d} {np.mean(vals):>+10.4f} {np.std(vals):>10.4f} "
              f"{np.percentile(vals, 5):>+10.4f} {np.percentile(vals, 95):>+10.4f}")

    # ── Analysis 6: Correlation matrix of remaining error with features ───
    print("\n" + "=" * 80)
    print("ANALYSIS 6: Correlation of remaining rdot error with state features")
    print("=" * 80)

    remaining = rdot_res_full_d
    features = {
        "vx": regime_vx,
        "vy": regime_vy,
        "r": regime_r,
        "|steer|": np.abs(regime_steer),
        "δ̇": regime_delta_rate,
        "|δ̇|": np.abs(regime_delta_rate),
        "δ̇²": regime_delta_rate ** 2,
        "δ̇·vx": regime_delta_rate * regime_vx,
        "steer_rate": regime_steer_rate,
        "|steer_rate|": np.abs(regime_steer_rate),
        "r·vx": regime_r * regime_vx,
        "vy·r": regime_vy * regime_r,
        "|r|·|δ̇|": np.abs(regime_r) * np.abs(regime_delta_rate),
    }

    print(f"{'Feature':>15s} {'Pearson r':>10s} {'|r|':>8s}")
    corrs = {}
    for name, feat in features.items():
        valid = np.isfinite(feat) & np.isfinite(remaining)
        if np.sum(valid) < 20:
            continue
        r_corr = float(np.corrcoef(feat[valid], remaining[valid])[0, 1])
        corrs[name] = r_corr
        print(f"{name:>15s} {r_corr:>+10.4f} {abs(r_corr):>8.4f}")

    # Sort by |correlation|
    print("\nTop features by |correlation|:")
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, r_corr in sorted_corrs[:5]:
        print(f"  {name:>15s}: r={r_corr:+.4f}")

    # ── Analysis 7: What would additional polynomial features help? ────────
    print("\n" + "=" * 80)
    print("ANALYSIS 7: Linear regression — how much variance can new features explain?")
    print("=" * 80)

    # Try fitting remaining error with candidate new features
    candidate_features = {
        "δ̇²": regime_delta_rate ** 2,
        "|δ̇|·sign(δ̇)·vx²": np.sign(regime_delta_rate) * regime_delta_rate ** 2 * regime_vx,
        "r²": regime_r ** 2,
        "r³": regime_r ** 3,
        "vy²·sign(vy)": np.sign(regime_vy) * regime_vy ** 2,
        "steer·vx²": regime_steer * regime_vx ** 2,
        "|δ̇|³": np.abs(regime_delta_rate) ** 3 * np.sign(regime_delta_rate),
        "r·δ̇": regime_r * regime_delta_rate,
    }

    # Current features (already in the polynomial)
    current_features = np.column_stack([
        np.ones(n_good),                         # w0
        regime_r,                                 # w1
        regime_r * regime_vx,                     # w2 (approx, using raw vx)
        regime_delta_rate,                        # w3
        regime_delta_rate * regime_vx,            # w4
        regime_vy,                                # w5
    ])

    # Error without polynomial (this is what the polynomial is trying to fix)
    target = rdot_res_nores_d

    # Fit current features
    from numpy.linalg import lstsq
    coefs_current, residuals_current, _, _ = lstsq(current_features, target, rcond=None)
    pred_current = current_features @ coefs_current
    ss_total = np.sum((target - np.mean(target)) ** 2)
    ss_res_current = np.sum((target - pred_current) ** 2)
    r2_current = 1 - ss_res_current / ss_total

    print(f"  Current 6 features R²: {r2_current:.4f}")
    print(f"  Current feature weights (OLS fit):")
    feat_names = ["bias", "r", "r·vx", "δ̇", "δ̇·vx", "vy"]
    for name, c in zip(feat_names, coefs_current):
        print(f"    {name:>10s}: {c:+.4f}")

    # Add candidate features one at a time
    print(f"\n  Adding one feature at a time to the current 6:")
    print(f"  {'Feature':>25s} {'R² new':>10s} {'ΔR²':>10s}")
    for name, feat in candidate_features.items():
        X_aug = np.column_stack([current_features, feat])
        coefs_aug, _, _, _ = lstsq(X_aug, target, rcond=None)
        pred_aug = X_aug @ coefs_aug
        ss_res_aug = np.sum((target - pred_aug) ** 2)
        r2_aug = 1 - ss_res_aug / ss_total
        delta_r2 = r2_aug - r2_current
        print(f"  {name:>25s} {r2_aug:>10.4f} {delta_r2:>+10.4f}")

    # ── Analysis 8: Rollout error accumulation — where does it happen? ────
    print("\n" + "=" * 80)
    print("ANALYSIS 8: 5s rollout error — early vs late accumulation")
    print("=" * 80)

    # Run 50 rollouts, track per-step yaw error
    rollout_steps = int(round(5.0 / args.dt))
    max_start = n - 1 - rollout_steps
    if max_start > 0:
        rng = np.random.default_rng(42)
        n_roll = min(50, max_start + 1)

        # Use test split for rollouts
        seg_test = select_split(raw, "test", args.train_ratio, 1)
        if args.max_seconds > 0:
            seg_test = truncate_duration(seg_test, args.max_seconds)
        n_test = len(seg_test["time"])
        max_start_test = n_test - 1 - rollout_steps
        if max_start_test > 0:
            starts = rng.choice(np.arange(max_start_test + 1), size=min(n_roll, max_start_test + 1), replace=False)
            n_actual = len(starts)

            err_yaw_per_step = np.zeros((n_actual, rollout_steps + 1))
            err_pos_per_step = np.zeros((n_actual, rollout_steps + 1))

            for ri, si in enumerate(starts):
                si = int(si)
                xs = build_state8(
                    seg_test["x"][si], seg_test["y"][si], seg_test["yaw"][si],
                    seg_test["vx"][si], seg_test["vy"][si], seg_test["r"][si],
                    seg_test["steer"][si], seg_test["throttle"][si],
                    p, steer_sign, "mapped")

                gt_yaw = seg_test["yaw"][si:si + rollout_steps + 1]
                gt_x = seg_test["x"][si:si + rollout_steps + 1]
                gt_y = seg_test["y"][si:si + rollout_steps + 1]

                pred = np.zeros((rollout_steps + 1, 8))
                pred[0] = xs
                for k in range(rollout_steps):
                    uk = map_controls(seg_test["steer"][si + k], seg_test["throttle"][si + k],
                                     p, steer_sign)
                    pred[k + 1] = rk4_step(pred[k], uk, args.dt, p, terrain_fn)

                err_yaw_per_step[ri] = np.degrees(np.abs(wrap_to_pi(pred[:, 2] - gt_yaw)))
                err_pos_per_step[ri] = np.hypot(pred[:, 0] - gt_x, pred[:, 1] - gt_y)

            # When does most error accumulate?
            time_axis = np.arange(rollout_steps + 1) * args.dt
            median_yaw = np.median(err_yaw_per_step, axis=0)
            # Compute error accumulation rate: d(error)/dt
            yaw_rate = np.diff(median_yaw) / args.dt

            print(f"  Yaw error accumulation rate (deg/s) at different horizons:")
            for t_s in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:
                idx = min(int(round(t_s / args.dt)), rollout_steps - 1)
                print(f"    t={t_s:.2f}s: {yaw_rate[idx]:.2f} deg/s (cumulative: {median_yaw[idx+1]:.2f} deg)")

            # Fraction of total error in first 1s vs last 4s
            idx_1s = int(round(1.0 / args.dt))
            err_first_1s = median_yaw[idx_1s]
            err_total = median_yaw[-1]
            print(f"\n  Error in first 1s: {err_first_1s:.2f} deg ({100*err_first_1s/max(err_total, 0.01):.1f}%)")
            print(f"  Error in last 4s:  {err_total - err_first_1s:.2f} deg ({100*(1-err_first_1s/max(err_total, 0.01)):.1f}%)")
            print(f"  Total at 5s:       {err_total:.2f} deg")

    # ── PLOTS ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(20, 16), constrained_layout=True)

    # Plot 1: δ̇ vs rdot error without w3,w4 (the error that w3 is trying to fix)
    ax = axes[0, 0]
    ax.scatter(regime_delta_rate, rdot_res_no_w3w4_d, s=1, alpha=0.1, c="tab:purple")
    # Overlay the linear fit (what w3+w4 does)
    dr_range = np.linspace(-5, 5, 100)
    ax.plot(dr_range, np.degrees((res_w3 * dr_range) / Iz), "r-", lw=2,
            label=f"w3·δ̇/Iz (w3={res_w3:.0f})")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.axvline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("δ̇ [rad/s]")
    ax.set_ylabel("rdot error w/o w3,w4 [deg/s²]")
    ax.set_title("Error that w3 is trying to fix")
    ax.legend(fontsize=8)
    ax.set_xlim(-5, 5)
    ax.grid(alpha=0.3)

    # Plot 2: remaining error (after full polynomial) vs δ̇
    ax = axes[0, 1]
    ax.scatter(regime_delta_rate, rdot_res_full_d, s=1, alpha=0.1, c="tab:red")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.axvline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("δ̇ [rad/s]")
    ax.set_ylabel("rdot error (full model) [deg/s²]")
    ax.set_title("Remaining error AFTER full polynomial")
    ax.set_xlim(-5, 5)
    ax.grid(alpha=0.3)

    # Plot 3: remaining error vs speed
    ax = axes[0, 2]
    ax.scatter(np.abs(regime_vx), rdot_res_full_d, s=1, alpha=0.1, c="tab:blue")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("|vx| [m/s]")
    ax.set_ylabel("rdot error (full model) [deg/s²]")
    ax.set_title("Remaining error vs speed")
    ax.grid(alpha=0.3)

    # Plot 4: histograms of rdot error - with vs without polynomial
    ax = axes[1, 0]
    bins = np.linspace(-100, 100, 200)
    ax.hist(rdot_res_nores_d, bins=bins, alpha=0.5, color="tab:red", label="No polynomial", density=True)
    ax.hist(rdot_res_full_d, bins=bins, alpha=0.5, color="tab:green", label="With polynomial", density=True)
    ax.set_xlabel("rdot residual [deg/s²]")
    ax.set_title("Polynomial residual effect on rdot error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 5: model rdot vs data rdot
    ax = axes[1, 1]
    lim = float(np.percentile(np.abs(np.degrees(data_rdot)), 98))
    ax.scatter(np.degrees(data_rdot), np.degrees(model_rdot_full), s=1, alpha=0.1, c="tab:green")
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.5, label="y=x")
    ax.set_xlabel("Data rdot [deg/s²]")
    ax.set_ylabel("Model rdot (full) [deg/s²]")
    ax.set_title("Model vs data yaw acceleration")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 6: w3 contribution vs remaining error
    ax = axes[1, 2]
    w3w4_correction = np.degrees((poly_w3_contrib + poly_w4_contrib) / Iz)
    ax.scatter(w3w4_correction, rdot_res_full_d, s=1, alpha=0.1, c="tab:orange")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.axvline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("w3+w4 correction applied [deg/s²]")
    ax.set_ylabel("Remaining rdot error [deg/s²]")
    ax.set_title("Correction applied vs remaining error")
    ax.grid(alpha=0.3)

    # Plot 7: rollout yaw error accumulation
    ax = axes[2, 0]
    if 'time_axis' in dir() and 'median_yaw' in dir():
        ax.plot(time_axis, median_yaw, "b-", lw=2, label="Median yaw error")
        ax.fill_between(time_axis,
                        np.percentile(err_yaw_per_step, 10, axis=0),
                        np.percentile(err_yaw_per_step, 90, axis=0),
                        alpha=0.2, color="blue", label="Q10-Q90")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Yaw error [deg]")
        ax.set_title("Yaw error accumulation (test rollouts)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Plot 8: yaw error rate
    ax = axes[2, 1]
    if 'yaw_rate' in dir():
        time_mid = time_axis[:-1] + 0.5 * args.dt
        ax.plot(time_mid, yaw_rate, "r-", lw=1.5)
        ax.axhline(np.mean(yaw_rate), color="k", ls="--", alpha=0.5,
                   label=f"mean={np.mean(yaw_rate):.2f} deg/s")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("d(yaw_error)/dt [deg/s]")
        ax.set_title("Yaw error accumulation RATE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Plot 9: polynomial Mz breakdown
    ax = axes[2, 2]
    poly_other = poly_Mz - poly_w3_contrib - poly_w4_contrib
    labels = ["w3·δ̇", "w4·δ̇·vx", "other (w0-2,w5)"]
    magnitudes = [
        float(np.mean(np.abs(poly_w3_contrib))),
        float(np.mean(np.abs(poly_w4_contrib))),
        float(np.mean(np.abs(poly_other))),
    ]
    ax.bar(labels, magnitudes, color=["tab:red", "tab:orange", "tab:blue"], alpha=0.7)
    ax.set_ylabel("Mean |Mz contribution| [N·m]")
    ax.set_title("Polynomial residual: mean |contribution| by term")
    ax.grid(alpha=0.3)

    fig.suptitle("Root Cause Analysis: Remaining Yaw Error", fontsize=14)
    fig.savefig(args.output_dir / "rootcause.png", dpi=180)
    plt.close(fig)

    # ── Save JSON ─────────────────────────────────────────────────────────
    payload = {
        "config": str(args.config),
        "split": args.split,
        "n_valid_steps": n_good,
        "rdot_res_std_nopoly_degps2": float(np.std(rdot_res_nores_d)),
        "rdot_res_std_full_degps2": float(np.std(rdot_res_full_d)),
        "rdot_res_std_no_w3w4_degps2": float(np.std(rdot_res_no_w3w4_d)),
        "r2_current_features": float(r2_current),
        "top_correlations": {name: float(r) for name, r in sorted_corrs[:5]},
    }
    (args.output_dir / "rootcause_results.json").write_text(json.dumps(payload, indent=2))

    print(f"\nPlots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
