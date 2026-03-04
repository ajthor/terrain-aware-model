#!/usr/bin/env python3
"""Step 0 — Delay diagnostic: cross-correlation between controls and vehicle response.

Estimates timing offset between:
  1. steering_input  ↔  yaw rate (r = ang_vel_z)
  2. throttle_input  ↔  longitudinal acceleration (finite-diff vx)

Uses TRAIN split only (no test leakage).
Outputs: console summary + correlation plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def load_data(data_root: Path, transmission: str, terrain: str):
    base = data_root / "data" / transmission / "train" / terrain
    odom = np.genfromtxt(base / "odom.csv", delimiter=",", names=True)
    ctrl = np.genfromtxt(base / "controls.csv", delimiter=",", names=True)
    n = min(len(odom), len(ctrl))
    odom, ctrl = odom[:n], ctrl[:n]
    return odom, ctrl, n


def cross_corr_normalized(x, y, max_lag):
    """Normalized cross-correlation for lags in [-max_lag, +max_lag].

    Returns (lags, corr) where positive lag means y leads x
    (i.e., x[t] correlates best with y[t - lag]).
    """
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        lags = np.arange(-max_lag, max_lag + 1)
        return lags, np.zeros_like(lags, dtype=float)

    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            corr[i] = np.sum(x[lag:] * y[:n - lag])
        else:
            corr[i] = np.sum(x[:n + lag] * y[-lag:])
    corr /= (n * sx * sy)
    return lags, corr


def find_peak(lags, corr):
    """Find lag of maximum absolute correlation."""
    idx = np.argmax(np.abs(corr))
    return int(lags[idx]), float(corr[idx])


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Diagnose control-state timing offset.")
    p.add_argument("--data-root", type=Path,
                    default=root / "function_encoder_beamng")
    p.add_argument("--transmission", default="m1")
    p.add_argument("--terrain", default="asphalt")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-lag-samples", type=int, default=40,
                    help="Max lag to search (in samples). At 50 Hz, 40 = ±0.8s.")
    p.add_argument("--output-dir", type=Path,
                    default=root / "results" / "delay_diagnostic")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    odom, ctrl, n = load_data(args.data_root, args.transmission, args.terrain)

    # Use train split only
    cut = int(args.train_ratio * n)
    odom = odom[:cut]
    ctrl = ctrl[:cut]
    n = cut

    # Extract signals as plain 1-D arrays (structured arrays don't slice well)
    t = np.array(odom["time"], dtype=float)
    dt_arr = np.diff(t)
    dt_median = float(np.median(dt_arr[dt_arr > 1e-6]))

    vx = np.array(odom["vel_x"], dtype=float)
    vy = np.array(odom["vel_y"], dtype=float)
    yaw_rate = np.array(odom["ang_vel_z"], dtype=float)
    steer = np.array(ctrl["steering_input"], dtype=float)
    throttle = np.array(ctrl["throttle_input"], dtype=float)

    # Compute finite-difference longitudinal acceleration
    ax_fd = np.gradient(vx, t)

    # Also compute yaw acceleration (finite diff of yaw rate)
    rdot_fd = np.gradient(yaw_rate, t)

    # Also check if controls.csv has its own time column and measure offset
    ctrl_t = ctrl["time"]
    time_offset_raw = ctrl_t - t
    time_offset_mean = float(np.mean(time_offset_raw))
    time_offset_std = float(np.std(time_offset_raw))

    print("=" * 70)
    print("DELAY DIAGNOSTIC — TRAIN SPLIT ONLY")
    print("=" * 70)
    print(f"Samples: {n}  |  dt_median: {dt_median:.6f}s  |  ~{1/dt_median:.1f} Hz")
    print(f"Train duration: {t[-1] - t[0]:.1f}s")
    print()

    # Report raw timestamp offset between odom and controls
    print("--- Raw timestamp alignment ---")
    print(f"ctrl_time - odom_time: mean={time_offset_mean:.6f}s, "
          f"std={time_offset_std:.6f}s, "
          f"min={float(np.min(time_offset_raw)):.6f}s, "
          f"max={float(np.max(time_offset_raw)):.6f}s")
    print()

    max_lag = args.max_lag_samples

    # ── Cross-correlations ──────────────────────────────────────────────

    pairs = [
        ("steer", "yaw_rate (r)", steer, yaw_rate),
        ("steer", "yaw_accel (rdot)", steer, rdot_fd),
        ("throttle", "ax (finite-diff vx)", throttle, ax_fd),
        ("throttle", "vx", throttle, vx),
        ("|steer|", "|r|", np.abs(steer), np.abs(yaw_rate)),
    ]

    results = []
    for ctrl_name, resp_name, ctrl_sig, resp_sig in pairs:
        lags, corr = cross_corr_normalized(ctrl_sig, resp_sig, max_lag)
        peak_lag, peak_corr = find_peak(lags, corr)
        peak_time_ms = peak_lag * dt_median * 1000
        results.append({
            "ctrl": ctrl_name,
            "resp": resp_name,
            "peak_lag_samples": peak_lag,
            "peak_time_ms": peak_time_ms,
            "peak_corr": peak_corr,
            "lags": lags,
            "corr": corr,
        })

    # Print results
    print("--- Cross-correlation peaks ---")
    print(f"{'Control':<14} {'Response':<22} {'Peak lag':>10} {'Time (ms)':>10} {'Corr':>8}")
    print("-" * 70)
    for res in results:
        print(f"{res['ctrl']:<14} {res['resp']:<22} {res['peak_lag_samples']:>10d} "
              f"{res['peak_time_ms']:>10.1f} {res['peak_corr']:>8.4f}")

    print()
    print("Interpretation:")
    print("  Positive lag = control LEADS response (expected: small positive = actuator delay)")
    print("  Negative lag = response LEADS control (suggests control timestamps are LATE)")
    print(f"  1 sample ≈ {dt_median*1000:.1f} ms")
    print()

    # ── Sliding-window analysis (check if delay varies with speed) ──────

    print("--- Delay vs speed regime ---")
    speed_quantiles = np.percentile(np.abs(vx), [0, 25, 50, 75, 100])
    print(f"Speed quantiles (|vx|): {speed_quantiles}")

    window = n // 4  # quarter-segment windows
    step = window // 2
    print(f"\nSliding window: {window} samples ({window*dt_median:.1f}s), step {step}")
    print(f"{'Window':>8} {'mean|vx|':>10} {'steer↔r lag':>14} {'thr↔ax lag':>12}")

    for wi, start in enumerate(range(0, n - window, step)):
        end = start + window
        seg_vx = np.abs(vx[start:end])

        _, c1 = cross_corr_normalized(steer[start:end], yaw_rate[start:end], max_lag)
        lag1, _ = find_peak(np.arange(-max_lag, max_lag + 1), c1)

        _, c2 = cross_corr_normalized(throttle[start:end], ax_fd[start:end], max_lag)
        lag2, _ = find_peak(np.arange(-max_lag, max_lag + 1), c2)

        print(f"{wi:>8d} {float(np.mean(seg_vx)):>10.2f} {lag1:>14d} {lag2:>12d}")

    # ── Also check: does shifting controls improve 1-step prediction? ──

    print("\n--- 1-step prediction error vs control shift ---")
    print("(Lower = better alignment)")
    print(f"{'Shift':>6} {'RMSE_ax':>10} {'RMSE_rdot':>10} {'RMSE_combined':>14}")

    for shift in range(-10, 11):
        if shift >= 0:
            s_ctrl = slice(shift, n - max(0, -shift) if -shift > 0 else n)
            s_resp = slice(0, n - shift)
        else:
            s_ctrl = slice(0, n + shift)
            s_resp = slice(-shift, n)

        # Ensure same length
        length = min(s_ctrl.stop - s_ctrl.start, s_resp.stop - s_resp.start)
        if length < 100:
            continue

        steer_s = steer[s_ctrl.start:s_ctrl.start + length]
        thr_s = throttle[s_ctrl.start:s_ctrl.start + length]
        ax_s = ax_fd[s_resp.start:s_resp.start + length]
        rdot_s = rdot_fd[s_resp.start:s_resp.start + length]

        # Simple linear proxy: how well does |steer| predict |rdot|?
        # And how well does throttle predict ax?
        rmse_ax = float(np.sqrt(np.mean((thr_s * 7.5 - ax_s) ** 2)))
        rmse_rdot = float(np.sqrt(np.mean((steer_s * 0.5 - rdot_s) ** 2)))
        rmse_combined = rmse_ax + rmse_rdot

        print(f"{shift:>6d} {rmse_ax:>10.4f} {rmse_rdot:>10.4f} {rmse_combined:>14.4f}")

    # ── Plots ────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.5 * len(results)),
                              constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        lag_times = res["lags"] * dt_median * 1000  # ms
        ax.plot(lag_times, res["corr"], "b-", lw=1.5)
        ax.axvline(res["peak_time_ms"], color="red", ls="--", lw=1,
                    label=f"peak={res['peak_lag_samples']} samples ({res['peak_time_ms']:.0f}ms), "
                          f"corr={res['peak_corr']:.3f}")
        ax.axvline(0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Lag [ms] (positive = control leads)")
        ax.set_ylabel("Normalized xcorr")
        ax.set_title(f"{res['ctrl']} → {res['resp']}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Control → Response Cross-Correlation (TRAIN split)", fontsize=13)
    plot_path = args.output_dir / "delay_cross_correlation.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")

    # ── Time-domain overlay plot ─────────────────────────────────────────

    # Show a short segment to visually check alignment
    seg_start = n // 3
    seg_len = min(500, n - seg_start)  # ~10s at 50Hz
    sl = slice(seg_start, seg_start + seg_len)
    t_seg = t[sl] - t[seg_start]

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True,
                                      sharex=True)

    ax1.plot(t_seg, steer[sl], "b-", lw=1, label="steer", alpha=0.8)
    ax1r = ax1.twinx()
    ax1r.plot(t_seg, yaw_rate[sl], "r-", lw=1, label="yaw rate (r)", alpha=0.8)
    ax1.set_ylabel("Steering input", color="b")
    ax1r.set_ylabel("Yaw rate [rad/s]", color="r")
    ax1.set_title("Steering vs Yaw Rate — look for phase offset")
    ax1.legend(loc="upper left", fontsize=9)
    ax1r.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(t_seg, throttle[sl], "b-", lw=1, label="throttle", alpha=0.8)
    ax2r = ax2.twinx()
    ax2r.plot(t_seg, ax_fd[sl], "r-", lw=1, label="ax (finite-diff)", alpha=0.8)
    ax2.set_ylabel("Throttle input", color="b")
    ax2r.set_ylabel("Longitudinal accel [m/s²]", color="r")
    ax2.set_xlabel("Time [s]")
    ax2.set_title("Throttle vs Longitudinal Acceleration — look for phase offset")
    ax2.legend(loc="upper left", fontsize=9)
    ax2r.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    fig2.suptitle("Time-domain overlay (10s segment from train)", fontsize=13)
    overlay_path = args.output_dir / "delay_time_overlay.png"
    fig2.savefig(overlay_path, dpi=180)
    plt.close(fig2)
    print(f"Overlay saved: {overlay_path}")

    # ── Summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    steer_r_result = results[0]  # steer ↔ yaw rate
    thr_ax_result = results[2]   # throttle ↔ ax

    print(f"Steering → Yaw rate:  peak lag = {steer_r_result['peak_lag_samples']} samples "
          f"= {steer_r_result['peak_time_ms']:.0f} ms  (corr={steer_r_result['peak_corr']:.3f})")
    print(f"Throttle → Long accel: peak lag = {thr_ax_result['peak_lag_samples']} samples "
          f"= {thr_ax_result['peak_time_ms']:.0f} ms  (corr={thr_ax_result['peak_corr']:.3f})")

    if abs(steer_r_result['peak_lag_samples']) <= 1 and abs(thr_ax_result['peak_lag_samples']) <= 1:
        print("\n→ Timing looks well-aligned (within ±1 sample). H1 likely NOT the main issue.")
    elif abs(steer_r_result['peak_lag_samples']) >= 2 or abs(thr_ax_result['peak_lag_samples']) >= 2:
        print(f"\n→ SIGNIFICANT timing offset detected. H1 is SUPPORTED.")
        print(f"  Recommended: apply delay correction before model tuning.")
    else:
        print("\n→ Marginal offset (~1 sample). May be partially due to actuator dynamics.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
