import numpy as np


def sat(x, lim):
    lim = abs(lim)
    return np.clip(x, -lim, lim)


def clip(x, lo, hi):
    return np.clip(x, lo, hi)


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

    ax_meas = float(meas.get("ax", 0.0))
    Fz_f = clip(0.5 * Fz_tot - (m * h_com / L) * ax_meas, 0.0, Fz_tot)
    Fz_r = Fz_tot - Fz_f

    ay_meas = float(meas.get("ay", 0.0))
    ay_norm = clip(np.abs(ay_meas) / g, 0.0, 1.0)
    Cf = Cf0 * (1.0 - k_lat_f * ay_norm)
    Cr = Cr0 * (1.0 - k_lat_r * ay_norm)

    alpha_f = delta - np.arctan2(vy + lf * r, vx_eff)
    alpha_r = -np.arctan2(vy - lr * r, vx_eff)

    Fy_f = sat(Cf * alpha_f, mu * Fz_f)
    Fy_r = sat(Cr * alpha_r, mu * Fz_r)
    Fx_r = sat(k_a * a, mu * Fz_r)

    cd = np.cos(delta)
    sd = np.sin(delta)
    Fx = -Fy_f * sd + Fx_r
    Fy = Fy_f * cd + Fy_r
    Mz = lf * (Fy_f * cd) - lr * Fy_r

    if residual is None:
        d_x = 0.0
        d_y = 0.0
        d_r = 0.0
    else:
        if x.ndim != 1:
            raise ValueError("residual is only supported for single-state inputs")
        d_x, d_y, d_r = residual(x, meas, phi)

    Xdot = vx * cpsi - vy * spsi
    Ydot = vx * spsi + vy * cpsi
    psidot = r

    vxdot = r * vy + (Fx + Fgx + d_x) / m
    vydot = -r * vx + (Fy + Fgy + d_y) / m
    rdot = (Mz + d_r) / Iz

    deltadot = (delta_cmd - delta) / tau_delta
    adot = (a_cmd - a) / tau_a

    deltadot = np.where((delta >= delta_max) & (deltadot > 0.0), 0.0, deltadot)
    deltadot = np.where((delta <= -delta_max) & (deltadot < 0.0), 0.0, deltadot)
    adot = np.where((a >= a_max) & (adot > 0.0), 0.0, adot)
    adot = np.where((a <= a_min) & (adot < 0.0), 0.0, adot)

    return np.stack((Xdot, Ydot, psidot, vxdot, vydot, rdot, deltadot, adot), axis=-1)
