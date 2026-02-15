#!/usr/bin/env python3
"""Tune analytical model parameters using Optuna on rollout error only.

Requires: pip install optuna  (for --sampler cmaes also: pip install cmaes)

Default tuned parameters: Cf0, Cr0, lf, lr, Iz (yaw/lateral-dynamics critical).
Objective: weighted sum of rollout RMSE position, rollout RMSE yaw, rollout final position.

Samplers: tpe (default, more exploration via n_startup_trials), cmaes (good at escaping
local minima), random (no learning, use as baseline or for very rough landscapes).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import RandomSampler, TPESampler

from examples.analytical_open_loop_rollouts import (
    build_state8,
    load_model_config,
    load_raw_aligned,
    map_controls,
    rk4_step,
    select_split,
    truncate_duration,
    wrap_to_pi,
)


def _trimmed_mean(x: np.ndarray, trim_ratio: float = 0.1) -> float:
    if x.size == 0:
        return float("nan")
    if x.size < 8:
        return float(np.mean(x))
    xs = np.sort(x)
    cut = int(trim_ratio * x.size)
    if cut <= 0 or 2 * cut >= x.size:
        return float(np.mean(xs))
    return float(np.mean(xs[cut:-cut]))


def _parse_tune_params(raw: str) -> list[str]:
    out = [s.strip() for s in raw.split(",") if s.strip()]
    if not out:
        raise ValueError("No tune params specified.")
    return out


def get_param_bounds(
    cfg_full: dict,
    base_p: dict[str, float],
    base_terrain: dict[str, float],
    tune_params: list[str],
    bound_scale: float,
) -> dict[str, tuple[float, float]]:
    """Bounds for each tuned param. Uses test_alternatives * bound_scale, or defaults."""
    alternatives = cfg_full.get("test_alternatives", {})
    defaults_lo_hi = {
        "Cf0": (20_000.0, 200_000.0),
        "Cr0": (20_000.0, 200_000.0),
        "lf": (0.9, 1.8),
        "lr": (0.9, 1.8),
        "Iz": (1500.0, 5000.0),
        "mu_effective": (0.5, 1.0),
    }
    bounds: dict[str, tuple[float, float]] = {}

    for name in tune_params:
        if name == "mu_effective":
            base_val = float(base_terrain.get("mu_effective", 1.0))
        else:
            base_val = float(base_p[name])

        alt = alternatives.get(name)
        if isinstance(alt, list) and len(alt) >= 2:
            vals = [float(v) for v in alt]
            lo, hi = min(vals), max(vals)
            mid = 0.5 * (lo + hi)
            span_lo = mid - lo
            span_hi = hi - mid
            lo = mid - bound_scale * span_lo
            hi = mid + bound_scale * span_hi
        else:
            if name in ("lf", "lr"):
                lo, hi = base_val * 0.7, base_val * 1.3
            else:
                lo, hi = defaults_lo_hi.get(name, (base_val * 0.7, base_val * 1.3))

        if name in ("Cf0", "Cr0", "Iz", "lf", "lr"):
            lo = max(lo, 1e-6)
        bounds[name] = (float(lo), float(hi))

    return bounds


def apply_candidate(
    base_p: dict[str, float],
    base_terrain: dict[str, float],
    candidate: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    p = dict(base_p)
    terrain = dict(base_terrain)
    for k, v in candidate.items():
        if k == "mu_effective":
            terrain["mu_effective"] = float(v)
        else:
            p[k] = float(v)
    return p, terrain


def evaluate_rollouts(
    candidate: dict[str, float],
    base_p: dict[str, float],
    base_terrain: dict[str, float],
    steer_sign: float,
    internal_init: str,
    seg: dict[str, np.ndarray],
    rollout_starts: np.ndarray,
    rollout_steps: int,
    rollout_dt: float,
    trim_ratio: float = 0.1,
) -> dict[str, float]:
    """Run N rollouts with the candidate params; return trimmed-mean rollout metrics."""
    p, terrain_cfg = apply_candidate(base_p, base_terrain, candidate)
    dhx = float(terrain_cfg.get("dhx_default", 0.0))
    dhy = float(terrain_cfg.get("dhy_default", 0.0))
    mu = float(terrain_cfg.get("mu_effective", 1.0))
    terrain_fn = lambda _x, _y: (dhx, dhy, mu, None)

    x, y, yaw = seg["x"], seg["y"], seg["yaw"]
    vx, vy, r = seg["vx"], seg["vy"], seg["r"]
    steer, throttle = seg["steer"], seg["throttle"]

    rollout_rmse_pos: list[float] = []
    rollout_rmse_yaw_deg: list[float] = []
    rollout_final_pos: list[float] = []

    for start in rollout_starts:
        start = int(start)
        xs = build_state8(
            float(x[start]), float(y[start]), float(yaw[start]),
            float(vx[start]), float(vy[start]), float(r[start]),
            float(steer[start]), float(throttle[start]),
            p, steer_sign, internal_init,
        )
        gt = np.column_stack([
            x[start : start + rollout_steps + 1],
            y[start : start + rollout_steps + 1],
            yaw[start : start + rollout_steps + 1],
            vx[start : start + rollout_steps + 1],
            vy[start : start + rollout_steps + 1],
            r[start : start + rollout_steps + 1],
        ])
        pred = np.zeros((rollout_steps + 1, 6))
        pred[0] = xs[:6]
        for j in range(rollout_steps):
            idx = start + j
            uk = map_controls(steer[idx], throttle[idx], p, steer_sign)
            xs = rk4_step(xs, uk, rollout_dt, p, terrain_fn=terrain_fn)
            pred[j + 1] = xs[:6]

        pos_err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
        yaw_err_deg = np.degrees(np.abs(wrap_to_pi(pred[:, 2] - gt[:, 2])))

        rollout_rmse_pos.append(float(np.sqrt(np.mean(pos_err * pos_err))))
        rollout_rmse_yaw_deg.append(float(np.sqrt(np.mean(yaw_err_deg * yaw_err_deg))))
        rollout_final_pos.append(float(pos_err[-1]))

    a_pos = np.array(rollout_rmse_pos)
    a_yaw = np.array(rollout_rmse_yaw_deg)
    a_final = np.array(rollout_final_pos)
    return {
        "rollout_rmse_pos_m_mean": _trimmed_mean(a_pos, trim_ratio),
        "rollout_rmse_yaw_deg_mean": _trimmed_mean(a_yaw, trim_ratio),
        "rollout_final_pos_m_mean": _trimmed_mean(a_final, trim_ratio),
    }


def update_config_with_candidate(config_path: Path, candidate: dict[str, float], out_path: Path) -> None:
    import yaml  # type: ignore
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")
    nominal = cfg.setdefault("nominal_params", {})
    optional = cfg.setdefault("optional_params", {})
    terrain = cfg.setdefault("terrain_params", {})

    for k, v in candidate.items():
        fv = float(v)
        if k == "mu_effective":
            terrain["mu_effective"] = fv
        elif isinstance(nominal, dict) and k in nominal:
            nominal[k] = fv
        elif isinstance(optional, dict) and k in optional:
            optional[k] = fv
        else:
            nominal[k] = fv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Tune analytical model (Optuna TPE, rollout-only).")
    a = p.add_argument
    a("--data-root", type=Path, default=root / "function_encoder_beamng")
    a("--transmission", default="m1")
    a("--terrain", default="asphalt")
    a("--train-ratio", type=float, default=0.8)
    a("--downsample", type=int, default=1)
    a("--max-train-seconds", type=float, default=250.0, help="<=0 disables.")
    a("--rollout-seconds", type=float, default=5.0)
    a("--rollout-dt", type=float, default=0.05)
    a("--n-rollouts", type=int, default=24)
    a("--max-dt-factor", type=float, default=3.0)
    a("--trials", type=int, default=1000, help="Optuna trials.")
    a("--sampler", choices=["tpe", "cmaes", "random"], default="tpe",
      help="tpe=TPE (default), cmaes=CMA-ES (escapes local minima), random=random search.")
    a("--n-startup-trials", type=int, default=100,
      help="Random trials before TPE exploits (only for --sampler tpe).")
    a("--seed", type=int, default=7)
    a("--internal-init", choices=["mapped", "zero"], default="mapped")
    a("--tune-params", default="Cf0,Cr0,lf,lr,Iz",
      help="Comma-separated parameter names.")
    a("--bound-scale", type=float, default=2.0,
      help="Scale factor to expand bounds from config.")
    a("--trim-ratio", type=float, default=0.1, help="Trim ratio for robust mean.")
    a("--w-roll-pos", type=float, default=1.0)
    a("--w-roll-yaw", type=float, default=1.2, help="Emphasize yaw (drives position drift).")
    a("--w-roll-final", type=float, default=0.8)
    a("--log-every", type=int, default=20)
    a("--config", type=Path, default=root / "configs" / "analytical_scintilla_asphalt.yaml")
    a("--output-dir", type=Path, default=root / "results" / "analytical_tuning")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.rollout_seconds <= 0.0 or args.rollout_dt <= 0.0:
        raise ValueError("rollout-seconds and rollout-dt must be > 0.")
    if args.trials < 1:
        raise ValueError("trials must be >= 1.")

    tune_params = _parse_tune_params(args.tune_params)
    import yaml  # type: ignore
    cfg_full = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg_full, dict):
        raise ValueError(f"Invalid config: {args.config}")

    base_p, base_terrain, steer_sign = load_model_config(args.config)
    bounds = get_param_bounds(cfg_full, base_p, base_terrain, tune_params, args.bound_scale)

    raw = load_raw_aligned(args.data_root, args.transmission, args.terrain)
    seg = select_split(raw, "train", args.train_ratio, max(1, args.downsample))
    if args.max_train_seconds > 0:
        seg = truncate_duration(seg, args.max_train_seconds)

    t = seg["time"]
    n = len(t)
    if n < 100:
        raise ValueError("Training segment too short.")

    dt_raw = np.diff(t)
    dt_pos = dt_raw[dt_raw > 1e-6]
    if dt_pos.size == 0:
        raise ValueError("No positive dt in segment.")
    dt_median = float(np.median(dt_pos))
    dt = np.where(dt_raw > 1e-6, dt_raw, dt_median)
    good_step = dt <= args.max_dt_factor * dt_median

    rollout_steps = int(round(args.rollout_seconds / args.rollout_dt))
    if rollout_steps < 2:
        raise ValueError("Rollout horizon too short.")
    max_start = n - 1 - rollout_steps
    if max_start < 0:
        raise ValueError("Not enough samples for rollout horizon.")

    valid_roll_starts = [s for s in range(max_start + 1) if np.all(good_step[s : s + rollout_steps])]
    if not valid_roll_starts:
        valid_roll_starts = list(range(max_start + 1))
    rng = np.random.default_rng(args.seed)
    n_roll = min(args.n_rollouts, len(valid_roll_starts))
    rollout_starts = rng.choice(np.array(valid_roll_starts), size=n_roll, replace=False)

    weights = {
        "rollout_rmse_pos_m_mean": args.w_roll_pos,
        "rollout_rmse_yaw_deg_mean": args.w_roll_yaw,
        "rollout_final_pos_m_mean": args.w_roll_final,
    }

    baseline_candidate = {
        k: (base_terrain["mu_effective"] if k == "mu_effective" else base_p[k])
        for k in tune_params
    }
    baseline_metrics = evaluate_rollouts(
        baseline_candidate, base_p, base_terrain, steer_sign, args.internal_init,
        seg, rollout_starts, rollout_steps, args.rollout_dt, args.trim_ratio,
    )
    baseline_cost = sum(weights[k] * baseline_metrics[k] for k in weights)

    def objective(trial: optuna.Trial) -> float:
        candidate = {}
        for name in tune_params:
            lo, hi = bounds[name]
            candidate[name] = trial.suggest_float(name, lo, hi)
        metrics = evaluate_rollouts(
            candidate, base_p, base_terrain, steer_sign, args.internal_init,
            seg, rollout_starts, rollout_steps, args.rollout_dt, args.trim_ratio,
        )
        cost = sum(weights[k] * metrics[k] for k in weights)
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        for k, v in candidate.items():
            trial.set_user_attr(f"param_{k}", v)
        return cost

    if args.sampler == "tpe":
        n_startup = min(max(1, args.n_startup_trials), args.trials)
        sampler = TPESampler(seed=args.seed, n_startup_trials=n_startup)
    elif args.sampler == "random":
        sampler = RandomSampler(seed=args.seed)
    else:  # cmaes
        try:
            from optuna.samplers import CmaEsSampler
        except ImportError as e:
            raise ImportError(
                "Optuna CMA-ES sampler requires the cmaes package. Install with: pip install cmaes"
            ) from e
        sampler = CmaEsSampler(seed=args.seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best_candidate = {k: study.best_trial.user_attrs[f"param_{k}"] for k in tune_params}
    best_metrics = {
        k: study.best_trial.user_attrs[k]
        for k in ("rollout_rmse_pos_m_mean", "rollout_rmse_yaw_deg_mean", "rollout_final_pos_m_mean")
    }
    best_cost = study.best_value

    trials_csv = args.output_dir / "trials.csv"
    rows = []
    baseline_row = {"trial_id": 0, "value": baseline_cost}
    baseline_row.update({f"param_{k}": v for k, v in baseline_candidate.items()})
    baseline_row.update(baseline_metrics)
    rows.append(baseline_row)
    for i, trial in enumerate(study.trials):
        row = {"trial_id": i + 1, "value": trial.value}
        row.update({k: trial.user_attrs[k] for k in trial.user_attrs})
        rows.append(row)
    if rows:
        with trials_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    best_cfg_path = args.output_dir / "best_config.yaml"
    update_config_with_candidate(args.config, best_candidate, best_cfg_path)

    configs_best_path = args.config.parent / f"{args.config.stem}_best.yaml"
    update_config_with_candidate(args.config, best_candidate, configs_best_path)

    payload = {
        "settings": {
            "data_root": str(args.data_root),
            "transmission": args.transmission,
            "terrain": args.terrain,
            "split_used_for_tuning": "train",
            "train_ratio": args.train_ratio,
            "downsample": max(1, args.downsample),
            "max_train_seconds": args.max_train_seconds,
            "segment_duration_seconds": float(t[-1] - t[0]),
            "n_segment_samples": n,
            "dt_median": dt_median,
            "max_dt_factor": args.max_dt_factor,
            "rollout_seconds": args.rollout_seconds,
            "rollout_dt": args.rollout_dt,
            "rollout_steps": rollout_steps,
            "n_rollouts": n_roll,
            "n_trials": args.trials,
            "sampler": args.sampler,
            "n_startup_trials": args.n_startup_trials if args.sampler == "tpe" else None,
            "seed": args.seed,
            "tune_params": tune_params,
            "bound_scale": args.bound_scale,
            "internal_init": args.internal_init,
            "steer_sign": float(steer_sign),
            "weights": weights,
        },
        "baseline": {
            "candidate": {k: float(v) for k, v in baseline_candidate.items()},
            "cost": float(baseline_cost),
            "metrics": {k: float(v) for k, v in baseline_metrics.items()},
        },
        "best": {
            "candidate": {k: float(v) for k, v in best_candidate.items()},
            "cost": float(best_cost),
            "metrics": {k: float(v) for k, v in best_metrics.items()},
            "improvement_ratio": float(best_cost / max(1e-9, baseline_cost)),
        },
        "files": {
            "trials_csv": trials_csv.name,
            "best_config": best_cfg_path.name,
            "best_config_in_configs": str(configs_best_path),
        },
    }
    summary_path = args.output_dir / "tuning_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nTuning complete.")
    print(f"Output: {args.output_dir}")
    print(f"Best config (configs/): {configs_best_path}")
    print(f"Baseline cost: {baseline_cost:.4f}  Best cost: {best_cost:.4f}")
    print(f"Improvement:   x{best_cost / max(1e-9, baseline_cost):.4f} (lower is better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
