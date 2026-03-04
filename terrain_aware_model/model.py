import numpy as np


def sat(x, lim):
    lim = abs(lim)
    return np.clip(x, -lim, lim)


def clip(x, lo, hi):
    return np.clip(x, lo, hi)


def pacejka_fy(alpha, C_stiffness, D_peak, C_shape):
    """Simplified Pacejka Magic Formula for lateral tire force.

    Fy = D * sin(C_shape * arctan(B * alpha))

    where B = C_stiffness / (C_shape * D) to match the cornering stiffness
    at alpha=0: dFy/dalpha|_{alpha=0} = C_stiffness.

    Parameters
    ----------
    alpha : slip angle [rad]
    C_stiffness : cornering stiffness at zero slip [N/rad] (= Cf0 or Cr0)
    D_peak : peak lateral force [N] (= mu * Fz)
    C_shape : shape factor (~1.0-1.6), controls where peak occurs
    """
    D_peak = np.maximum(D_peak, 1.0)
    B = C_stiffness / np.maximum(C_shape * D_peak, 1.0)
    return D_peak * np.sin(C_shape * np.arctan(B * alpha))


def vehicle_dynamics(t, x, u, p, terrain=None, residual=None, meas=None):
    """
    Continuous-time dynamics for:
      x = [X, Y, psi, vx, vy, r, delta, a]
      u = [delta_cmd, a_cmd]

    p: dict with keys
      m, Iz, lf, lr, L, h_com, g
      Cf0, Cr0, k_a
      tau_delta, tau_a
      delta_max, a_min, a_max
      (optional) k_lat_f, k_lat_r

    terrain(X,Y) -> (dhx, dhy, mu, phi)
      dhx, dhy: terrain gradient components
      mu: effective friction coefficient at (X,Y)
      phi: extra terrain features (can be None)

    residual(x, meas, phi) -> (d_x, d_y, d_r)
      d_x, d_y in N, d_r in N*m

    meas: dict with optional keys "ax", "ay" for load-transfer / stiffness scaling.
    """
    if meas is None:
        meas = {}

    if not isinstance(x, np.ndarray) or not isinstance(u, np.ndarray):
        raise TypeError("x and u must be numpy arrays")

    X = x[..., 0]
    Y = x[..., 1]
    psi = x[..., 2]
    vx = x[..., 3]
    vy = x[..., 4]
    r = x[..., 5]
    delta = x[..., 6]
    a = x[..., 7]
    delta_cmd = u[..., 0]
    a_cmd = u[..., 1]

    m = p["m"]
    Iz = p["Iz"]
    lf = p["lf"]
    lr = p["lr"]
    L = p["L"]
    h_com = p["h_com"]
    g = p["g"]
    Cf0 = p["Cf0"]
    Cr0 = p["Cr0"]
    k_a = p["k_a"]
    tau_delta = max(float(p["tau_delta"]), 1e-3)
    tau_a = max(float(p["tau_a"]), 1e-3)
    delta_max = p["delta_max"]
    a_min = p["a_min"]
    a_max = p["a_max"]
    k_lat_f = float(p.get("k_lat_f", 0.0))
    k_lat_r = float(p.get("k_lat_r", 0.0))
    coast_c0 = float(p.get("coast_c0", 0.0))
    coast_c1 = float(p.get("coast_c1", 0.0))
    Cf_shape = float(p.get("Cf_shape", 0.0))
    Cr_shape = float(p.get("Cr_shape", 0.0))
    use_pacejka = Cf_shape > 0.0 and Cr_shape > 0.0

    # Polynomial Mz residual weights (rollout-trained via Optuna).
    # Each feature captures physics NOT in the analytical bicycle model:
    #   w0:  bias (constant yaw moment offset)
    #   w1:  yaw damping ∝ r  (roll/suspension effects have no r→Mz path)
    #   w2:  speed-dep yaw damping ∝ r·vx  (roll effects scale with speed)
    #   w3:  steering transient ∝ δ̇  (tire relaxation at low speed)
    #   w4:  steering transient ∝ δ̇·vx  (tire relaxation ∝ σ/vx)
    #   w5:  sideslip correction ∝ vy  (unmodeled vy→Mz coupling)
    #   w6:  speed-dep steering ∝ δ_cmd·vx²  (understeer/oversteer vs speed)
    #   w7:  nonlinear sideslip ∝ vy·|vy|  (asymmetric vy→Mz at large slip)
    res_w0 = float(p.get("res_w0", 0.0))
    res_w1 = float(p.get("res_w1", 0.0))
    res_w2 = float(p.get("res_w2", 0.0))
    res_w3 = float(p.get("res_w3", 0.0))
    res_w4 = float(p.get("res_w4", 0.0))
    res_w5 = float(p.get("res_w5", 0.0))
    res_w6 = float(p.get("res_w6", 0.0))
    res_w7 = float(p.get("res_w7", 0.0))

    vx_eff = np.where(np.abs(vx) > 0.2, vx, np.where(vx >= 0.0, 0.2, -0.2))

    if terrain is None:
        dhx = 0.0
        dhy = 0.0
        mu = 1.0
        phi = None
    else:
        if x.ndim != 1:
            raise ValueError("terrain is only supported for single-state inputs")
        out = terrain(float(X), float(Y))
        if len(out) == 3:
            dhx, dhy, mu = out
            phi = None
        else:
            dhx, dhy, mu, phi = out

    mu = np.maximum(mu, 0.05)

    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    s_slope = dhx * cpsi + dhy * spsi
    c_slope = -dhx * spsi + dhy * cpsi

    Fgx = -m * g * s_slope
    Fgy = -m * g * c_slope

    Fz_tot = m * g / np.sqrt(1.0 + dhx * dhx + dhy * dhy)

    # Longitudinal acceleration for load transfer:
    # Use measured ax if available (e.g. from IMU), otherwise estimate from
    # the model's own drive force state: ax ≈ k_a * a / m.
    if "ax" in meas:
        ax_lt = float(meas["ax"])
    else:
        ax_lt = k_a * a / m

    # Static weight split uses lr/L (not 50/50) — rear-biased for rear-drive.
    Fz_f = clip((lr / L) * Fz_tot - (m * h_com / L) * ax_lt, 0.0, Fz_tot)
    Fz_r = Fz_tot - Fz_f

    # Lateral acceleration for cornering stiffness reduction:
    # Use measured ay if available, otherwise estimate from model state.
    if "ay" in meas:
        ay_lat = float(meas["ay"])
    else:
        ay_lat = vx * r  # body-frame lateral acceleration ≈ centripetal
    ay_norm = clip(np.abs(ay_lat) / g, 0.0, 1.0)
    Cf = Cf0 * (1.0 - k_lat_f * ay_norm)
    Cr = Cr0 * (1.0 - k_lat_r * ay_norm)

    # Slip angles.
    alpha_f = delta - np.arctan2(vy + lf * r, vx_eff)
    alpha_r = -np.arctan2(vy - lr * r, vx_eff)

    # Longitudinal force: drive + coast/engine braking (computed before lateral
    # forces so we can apply the friction circle to Fy).
    Fx_drive = k_a * a
    throttle_frac = clip(a / np.where(a_max > 0.01, a_max, 0.01), 0.0, 1.0)
    Fx_coast = -(coast_c0 + coast_c1 * np.abs(vx)) * (1.0 - throttle_frac) * np.sign(vx_eff)
    Fx_r = sat(Fx_drive + Fx_coast, mu * Fz_r)

    # Combined slip: friction circle limits available lateral force.
    # Front axle sees lateral component of Fx through steering: Fx_f_lat ≈ 0
    # (front is not driven), so front peak is unchanged.
    # Rear axle carries all drive/brake force Fx_r.
    D_peak_f = mu * Fz_f
    D_peak_r = np.sqrt(np.maximum((mu * Fz_r) ** 2 - Fx_r ** 2, 0.0))

    if use_pacejka:
        Fy_f = pacejka_fy(alpha_f, Cf, D_peak_f, Cf_shape)
        Fy_r = pacejka_fy(alpha_r, Cr, D_peak_r, Cr_shape)
    else:
        Fy_f = sat(Cf * alpha_f, D_peak_f)
        Fy_r = sat(Cr * alpha_r, D_peak_r)

    cd = np.cos(delta)
    sd = np.sin(delta)
    Fx = -Fy_f * sd + Fx_r
    Fy = Fy_f * cd + Fy_r
    Mz = lf * (Fy_f * cd) - lr * Fy_r

    # Polynomial Mz residual: compute correction from state features.
    delta_rate = (delta_cmd - delta) / tau_delta
    d_r_poly = (res_w0
                + res_w1 * r
                + res_w2 * r * vx_eff
                + res_w3 * delta_rate
                + res_w4 * delta_rate * vx_eff
                + res_w5 * vy
                + res_w6 * delta_cmd * vx_eff ** 2
                + res_w7 * vy * np.abs(vy))

    if residual is None:
        d_x = 0.0
        d_y = 0.0
        d_r = d_r_poly
    else:
        if x.ndim != 1:
            raise ValueError("residual is only supported for single-state inputs")
        d_x, d_y, d_r = residual(x, meas, phi)
        d_r = d_r + d_r_poly

    Xdot = vx * cpsi - vy * spsi
    Ydot = vx * spsi + vy * cpsi
    psidot = r

    vxdot = r * vy + (Fx + Fgx + d_x) / m
    vydot = -r * vx + (Fy + Fgy + d_y) / m
    rdot = (Mz + d_r) / Iz

    deltadot = delta_rate
    adot = (a_cmd - a) / tau_a

    deltadot = np.where((delta >= delta_max) & (deltadot > 0.0), 0.0, deltadot)
    deltadot = np.where((delta <= -delta_max) & (deltadot < 0.0), 0.0, deltadot)
    adot = np.where((a >= a_max) & (adot > 0.0), 0.0, adot)
    adot = np.where((a <= a_min) & (adot < 0.0), 0.0, adot)

    return np.stack((Xdot, Ydot, psidot, vxdot, vydot, rdot, deltadot, adot),
                     axis=-1)
