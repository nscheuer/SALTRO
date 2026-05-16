"""Spike detection and removal for ALTRO trajectory optimization.

After each accepted iLQR forward-pass step, this module:
  1. Detects mid-trajectory "spikes" (homotopy artifacts in SO(3))
  2. Filters out physics-limited cases (actuator saturated & fighting correctly)
  3. Validates via cost comparison and keep-out check
  4. Substitutes a simple PD-controller segment over the spike window
  5. Re-rolls out the tail with iLQR gain correction for dynamic feasibility

All APIs are pure Python, using saltro_py bindings only.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Quaternion math helpers
# ---------------------------------------------------------------------------

def _quat_conj(q):
    """Conjugate of unit quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mult(q1, q2):
    """Hamilton product q1 * q2, scalar-first convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_error(q_target, q_current):
    """Error quaternion: q_err = q_target^{-1} * q_current, shortest path."""
    q_err = _quat_mult(_quat_conj(q_target), q_current)
    if q_err[0] < 0.0:
        q_err = -q_err
    return q_err


def _quat_angle(q_err):
    """Geodesic angle from error quaternion [w, x, y, z]."""
    w = float(np.clip(q_err[0], -1.0, 1.0))
    return 2.0 * np.arccos(abs(w))


def _rotation_matrix(q):
    """Body-to-inertial rotation matrix from unit quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _quat_to_mrp(q_err):
    """Modified Rodrigues Parameters from error quaternion (scalar-first)."""
    w = q_err[0]
    vec = q_err[1:]
    denom = 1.0 + w
    if abs(denom) < 1e-10:
        return vec / 1e-10
    return vec / denom


# ---------------------------------------------------------------------------
# RK4 integration helper
# ---------------------------------------------------------------------------

def _rk4_step(satellite, x, u, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k):
    """One RK4 step using satellite.dynamics."""
    def f(x_):
        return np.asarray(satellite.dynamics(x_, u, dist_cfg, R_k, B_k, S_k, V_k, rho_k))

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # Normalize quaternion
    q = x_next[3:7]
    qn = np.linalg.norm(q)
    if qn > 1e-10:
        x_next[3:7] = q / qn
    return x_next


# ---------------------------------------------------------------------------
# Phase 1: Spike detection
# ---------------------------------------------------------------------------

def _pointing_error(X, attitude_target, boresight, k):
    """Scalar pointing error at knot k."""
    target = attitude_target[:, k]
    x = X[:, k]
    q = x[3:7]

    if np.isnan(target[0]):
        # Vector pointing mode: angle between body boresight and target vector
        target_vec = target[1:4]
        norm = np.linalg.norm(target_vec)
        if norm < 1e-10:
            return 0.0
        target_vec = target_vec / norm
        C = _rotation_matrix(q)
        boresight_k = boresight[:, k]
        boresight_eci = C @ boresight_k
        boresight_eci /= max(np.linalg.norm(boresight_eci), 1e-10)
        return float(np.arccos(np.clip(np.dot(boresight_eci, target_vec), -1.0, 1.0)))
    else:
        # Quaternion mode
        q_err = _quat_error(target, q)
        return _quat_angle(q_err)


def _find_goal_transitions(attitude_target):
    """Return set of knot indices where goal changes."""
    N = attitude_target.shape[1]
    transitions = set()
    for k in range(1, N):
        if not np.allclose(attitude_target[:, k], attitude_target[:, k-1], atol=1e-9):
            transitions.add(k)
    return transitions


def _is_saturated(u, satellite, control_limit_scale, ratio_of_al_ceiling=0.9):
    """True if any dominant control channel is saturated against the
    effective AL-imposed ceiling (u_max × control_limit_scale).

    AL drives |u| to at most control_limit_scale · u_max (typically 0.75).
    Measuring saturation as a fraction of the hardware u_max was dead
    code — AL never let u reach 95% of u_max.  We now scale by
    control_limit_scale so the threshold tracks whatever the user sets.
    `ratio_of_al_ceiling` defaults to 0.95 = "within 5% of the effective
    AL ceiling" ≈ saturated.
    """
    threshold = ratio_of_al_ceiling * control_limit_scale
    n_mtq = satellite.numMTQ
    n_rw = satellite.numRW
    for i in range(n_mtq):
        mtq = satellite.getMTQ(i)
        u_max = float(mtq.u_max)
        if u_max > 0 and abs(u[i]) >= threshold * u_max:
            return True
    for i in range(n_rw):
        rw = satellite.getRW(i)
        u_max = float(rw.u_max)
        if u_max > 0 and abs(u[n_mtq + i]) >= threshold * u_max:
            return True
    return False


def _torque_opposes_error(x, tau_act, attitude_target_k, satellite):
    """True if actuator torque is driving angular velocity toward the target attitude.

    Checks: does tau_act have a component along the error rotation axis?
    Uses J^{-1} @ tau_act to get expected angular acceleration direction.
    """
    q = x[3:7]
    target = attitude_target_k

    if np.isnan(target[0]):
        return False  # Vector-pointing mode: defer to cost comparison

    q_err = _quat_error(target, q)
    err_axis = q_err[1:4]
    err_norm = np.linalg.norm(err_axis)
    if err_norm < 1e-8:
        return False  # Already at target, no axis to oppose

    n_err = err_axis / err_norm  # error rotation axis (body frame)

    # Expected angular acceleration from actuator torque
    J_inv = np.asarray(satellite.invInertiaNoRW)
    alpha = J_inv @ np.asarray(tau_act)  # 3-vector

    # If alpha opposes the error axis (dot < 0), torque is correcting the error
    return float(np.dot(alpha, n_err)) < 0.0


def detect_spikes(
    X,
    U,
    attitude_target,
    boresight,
    B,
    satellite,
    cnst_cfg,
    goal_switch_buffer: int = 15,
    min_consecutive: int = 5,
    exit_fudge: float = 2.0,
    min_prior_decrease_knots: int = 10,
    min_spike_ratio: float = 2.0,
    min_peak_rad: float = 1.0,
    entry_error_max_rad: float = 0.5,
    entry_gate_traj_min_factor: float = 1.5,
    prior_low_max_rad: float = 0.15,
    post_low_max_rad: float = 0.15,
    min_post_stable_knots: int = 10,
    post_vs_prior_ratio: float = 0.5,
    peak_exit_ratio: float = 0.3,
    signflip_override_post_min_max_rad: float = 0.5,
    verbose: bool = False,
) -> list:
    """Detect spike candidate windows in a trajectory.

    Parameters
    ----------
    X : (nx, N) trajectory states
    U : (nu, N-1) trajectory controls
    attitude_target : (4, N) goal quaternions (NaN in row 0 = vector-pointing)
    boresight : (3, N) body-frame boresight per knot
    B : (3, N) ECI magnetic field per knot
    satellite : saltro_py.Satellite
    cnst_cfg : ConstraintConfig
    goal_switch_buffer : knots to skip around goal transitions
    min_consecutive : minimum consecutive increasing-error knots to flag
    exit_fudge : exit when error <= entry_error * exit_fudge
    min_prior_decrease_knots : the error must have been decreasing for at least
        this many knots before the spike onset (ensures the trajectory was
        converging before the reversal, not just ramping up)
    min_spike_ratio : the peak error in the spike run must be at least this
        multiple of the entry error (ensures the jump is sudden and large,
        not just a small oscillation during the initial approach)

    Returns
    -------
    list of (t_enter, t_exit) tuples
    """
    N = X.shape[1]
    transitions = _find_goal_transitions(attitude_target)

    # Buffer zone: any knot within goal_switch_buffer of a transition
    buffered = set()
    for t in transitions:
        for dk in range(-goal_switch_buffer, goal_switch_buffer + 1):
            buffered.add(t + dk)

    # Compute error metric for each knot
    theta = np.zeros(N)
    for k in range(N):
        if k in buffered:
            theta[k] = np.nan
        else:
            theta[k] = _pointing_error(X, attitude_target, boresight, k)

    # --- Transition-based detection ---
    # The spike IS the hemisphere transition: where q · q_0 crosses zero.
    # The trajectory can't slew infinitely fast, so the transition takes a
    # few knots — during those knots PE peaks because the trajectory is
    # near the antipode region.
    # Detection model:
    #   1. Find each transition: knot where sign(q·q_0) flips relative to
    #      the previous knot.  No persistence required — even single-knot
    #      transitions count (a real fast spike).
    #   2. For each transition, find the local PE peak within ±max_half
    #      window.  Apply min_peak_rad filter.
    #   3. Walk PE outward from the peak to find spike window boundaries:
    #      left along the rising flank, right along the falling flank,
    #      capped at max_window_width total.
    #   4. Skip windows that extend to the trajectory end (legitimate
    #      slew tail in opposite hemisphere).
    #   5. Merge overlapping/adjacent windows.
    # No q_goal needed (q_0 reference is always defined, works for vector-
    # pointing mode too).
    q_ref = X[3:7, 0]

    dots = np.zeros(N)
    for kk in range(N):
        dots[kk] = float(np.dot(X[3:7, kk], q_ref))

    # Identify each transition: knot where sign(dot) differs from previous knot.
    transitions = []
    for kk in range(1, N):
        prev_sign = 1 if dots[kk - 1] >= 0 else -1
        cur_sign = 1 if dots[kk] >= 0 else -1
        if prev_sign != cur_sign:
            # Use the knot where |dot| is closer to zero as the transition.
            t_event = kk if abs(dots[kk]) < abs(dots[kk - 1]) else kk - 1
            transitions.append(t_event)

    if verbose and transitions:
        print(f"  [detect] hemisphere transitions: {transitions}")

    max_window_width = 20
    max_half = max_window_width // 2

    raw_candidates = []
    for k_trans in transitions:
        if k_trans in buffered:
            continue

        # Find local peak within ±max_half of the transition.
        lo = max(0, k_trans - max_half)
        hi = min(N - 1, k_trans + max_half)
        window_slice = theta[lo:hi + 1]
        if np.all(np.isnan(window_slice)):
            continue
        k_peak = lo + int(np.nanargmax(window_slice))
        peak_val = float(theta[k_peak])

        if peak_val < min_peak_rad:
            if verbose:
                print(f"  [detect] reject transition @{k_trans}: peak={np.degrees(peak_val):.1f}° < {np.degrees(min_peak_rad):.1f}°")
            continue

        # Walk left from k_peak along the rising flank (PE rises as we move
        # forward in time, so theta[k] < theta[k+1] when k is to the left
        # of the peak in the rising flank).
        t_enter = k_peak
        for i in range(1, max_half + 1):
            k_walk = k_peak - i
            if k_walk < 0 or np.isnan(theta[k_walk]):
                break
            if theta[k_walk] < theta[k_walk + 1]:
                t_enter = k_walk
            else:
                break

        # Walk right from k_peak along the falling flank (PE falls as we
        # move forward, so theta[k] < theta[k-1] on the right of the peak).
        t_exit = k_peak
        for i in range(1, max_half + 1):
            k_walk = k_peak + i
            if k_walk >= N or np.isnan(theta[k_walk]):
                break
            if theta[k_walk] < theta[k_walk - 1]:
                t_exit = k_walk
            else:
                break

        # Enforce total window cap (peak walk may exceed max_window_width
        # when both flanks are long; trim symmetrically toward k_peak).
        if t_exit - t_enter > max_window_width:
            overflow = (t_exit - t_enter) - max_window_width
            half_over = overflow // 2
            t_enter += half_over
            t_exit -= (overflow - half_over)

        # Skip windows that extend to the trajectory end — those are
        # legitimate slew tails ending in the opposite hemisphere relative
        # to q_0, not homotopy spikes.
        if t_exit >= N - 1:
            if verbose:
                print(f"  [detect] reject transition @{k_trans}: window ({t_enter},{t_exit}) ends at N-1 (legit slew)")
            continue

        if verbose:
            print(f"  [detect] accept transition @{k_trans}: window ({t_enter},{t_exit}) peak={np.degrees(peak_val):.1f}° (k_peak={k_peak})")
        raw_candidates.append((t_enter, t_exit))

    # --- Merge overlapping or adjacent windows ---
    raw_candidates.sort()
    merged = []
    for c_start, c_end in raw_candidates:
        if merged and c_start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], c_end))
        else:
            merged.append((c_start, c_end))

    if verbose and len(merged) != len(raw_candidates):
        print(f"  [detect] merged {len(raw_candidates)} raw → {len(merged)} candidates: {merged}")

    candidates = []
    # Actuation-driven filter: drop windows where the actuator is saturated
    # AND opposing the error.  That's physics-limited oscillation, not a
    # homotopy artifact worth substituting.
    for t_enter, t_exit in merged:
        mid = (t_enter + t_exit) // 2
        check_knots = [t_enter, mid, min(mid + (t_exit - t_enter) // 4, t_exit - 1)]
        check_knots = [ck for ck in check_knots if 0 <= ck < U.shape[1]]

        physics_limited_votes = 0
        for ck in check_knots:
            u_k = U[:, ck]
            x_k = X[:, ck]
            B_k = B[:, ck]
            tau_act = np.asarray(satellite.actuatorTorque(x_k, u_k, B_k))
            if _is_saturated(u_k, satellite, cnst_cfg.control_limit_scale) and _torque_opposes_error(
                x_k, tau_act, attitude_target[:, ck], satellite
            ):
                physics_limited_votes += 1

        if physics_limited_votes >= max(1, len(check_knots) // 2 + 1):
            if verbose:
                print(f"  [detect] reject ({t_enter},{t_exit}): physics-limited ({physics_limited_votes}/{len(check_knots)})")
            continue

        candidates.append((t_enter, t_exit))

    return candidates


# ---------------------------------------------------------------------------
# Phase 2: PD segment simulation
# ---------------------------------------------------------------------------

def _build_pd_control(x, q_target, satellite, B_eci, kp_q, kd_w, rw_scale=0.0):
    """Compute a simple PD control toward q_target.

    MTQ: least-norm dipole to achieve desired torque projected onto B-perp plane.
    RW: direct torque (remainder after MTQ), scaled by rw_scale.

    rw_scale=0.0 means MTQ-only (default) — keeps RW momentum near zero
    and avoids violating the RW momentum constraint in the AL outer loop.

    Returns control vector u of length (numMTQ + numRW).
    """
    q = x[3:7]
    omega = x[0:3]

    # Quaternion error
    q_err = _quat_error(q_target, q)

    # Desired torque (body frame) — gentle gains to avoid omega constraint violation
    tau_des = -kp_q * q_err[1:4] - kd_w * omega

    n_mtq = satellite.numMTQ
    n_rw = satellite.numRW
    u = np.zeros(n_mtq + n_rw)

    # --- MTQ contribution ---
    # Body-frame B: B_body = C^T @ B_eci  (C = body-to-ECI rotation)
    C = _rotation_matrix(q)          # body->ECI
    B_body = C.T @ B_eci             # ECI->body

    B_norm_sq = np.dot(B_body, B_body)
    tau_mtq_total = np.zeros(3)

    if B_norm_sq > 1e-20 and n_mtq > 0:
        A = np.zeros((3, n_mtq))
        for i in range(n_mtq):
            axis_i = np.asarray(satellite.getMTQ(i).axis)
            A[:, i] = np.cross(axis_i, B_body)

        AtA = A.T @ A
        try:
            m_raw = np.linalg.lstsq(AtA, A.T @ tau_des, rcond=None)[0]
        except np.linalg.LinAlgError:
            m_raw = np.zeros(n_mtq)

        for i in range(n_mtq):
            mtq = satellite.getMTQ(i)
            u_max = float(mtq.u_max)
            u[i] = float(np.clip(m_raw[i], -u_max, u_max))
            axis_i = np.asarray(satellite.getMTQ(i).axis)
            tau_mtq_total += u[i] * np.cross(axis_i, B_body)

    # --- RW contribution: scaled remainder (default 0 = MTQ-only) ---
    if rw_scale > 0.0 and n_rw > 0:
        tau_remaining = tau_des - tau_mtq_total
        for i in range(n_rw):
            rw = satellite.getRW(i)
            rw_axis = np.asarray(rw.axis)
            u_max = float(rw.u_max)
            tau_rw_i = float(np.dot(tau_remaining, rw_axis)) * rw_scale
            u[n_mtq + i] = float(np.clip(tau_rw_i, -u_max, u_max))

    return u


def simulate_pd_segment(
    x_start,
    x_target,
    n_steps,
    B_cols,
    S_cols,
    R_cols,
    V_cols,
    rho_cols,
    satellite,
    dist_cfg,
    dt,
    kp_q=2.0,
    kd_w=5.0,
    omega_max=None,
    h_max=None,
    rw_scale=0.0,
):
    """Simulate a PD-controlled trajectory from x_start toward x_target.

    Parameters
    ----------
    x_start : initial state at t_enter
    x_target : target state (X[:, t_exit]) — spike exit state
    n_steps : t_exit - t_enter
    B_cols, S_cols, R_cols, V_cols, rho_cols : (3/1, n_steps) environment slices
    satellite, dist_cfg, dt : dynamics configuration
    kp_q, kd_w : PD gains (tune per application)
    omega_max : if set, clamp ||omega|| to this value after each step (rad/s)
    h_max : if set, clamp |h_rw| to this value after each step (N·m·s)

    Returns
    -------
    X_pd : (nx, n_steps+1)
    U_pd : (nu, n_steps)
    """
    nx = len(x_start)
    n_mtq = satellite.numMTQ
    n_rw = satellite.numRW
    nu = n_mtq + n_rw

    q_target = x_target[3:7]

    X_pd = np.zeros((nx, n_steps + 1))
    U_pd = np.zeros((nu, n_steps))
    X_pd[:, 0] = x_start

    for k in range(n_steps):
        x_k = X_pd[:, k]
        B_k = B_cols[:, k]
        S_k = S_cols[:, k] if S_cols.ndim == 2 else S_cols
        R_k = R_cols[:, k] if R_cols.ndim == 2 else R_cols
        V_k = V_cols[:, k] if V_cols.ndim == 2 else V_cols
        rho_k = int(np.round(float(rho_cols[0, k]))) if rho_cols.ndim == 2 else int(rho_cols[k])

        u_k = _build_pd_control(x_k, q_target, satellite, B_k, kp_q, kd_w, rw_scale=rw_scale)
        U_pd[:, k] = u_k

        x_next = _rk4_step(satellite, x_k, u_k, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k)

        # Clamp angular velocity magnitude (prevents AL constraint violations)
        if omega_max is not None:
            omega_norm = np.linalg.norm(x_next[0:3])
            if omega_norm > omega_max:
                x_next[0:3] *= omega_max / omega_norm

        # Clamp RW momentum magnitude
        if h_max is not None:
            for i in range(n_rw):
                x_next[7 + i] = float(np.clip(x_next[7 + i], -h_max, h_max))

        X_pd[:, k + 1] = x_next

    return X_pd, U_pd


# ---------------------------------------------------------------------------
# Phase 3: Cost comparison
# ---------------------------------------------------------------------------

def compare_costs(
    X_orig,
    U_orig,
    X_pd,
    U_pd,
    t_enter,
    t_exit,
    satellite,
    B,
    boresight,
    attitude_target,
    cost_cfg,
    N,
):
    """Return True if PD segment has lower stage cost than original over [t_enter, t_exit).

    For no-goal segments, extends window 15% into the next goal segment using
    the original tail trajectory.
    """
    window = range(t_enter, t_exit)

    # Detect if this is a no-goal segment (all NaN attitude targets in window)
    all_no_goal = all(
        np.isnan(attitude_target[0, k]) for k in window
    )

    if all_no_goal:
        # Extend into first 15% of next goal segment
        transitions = _find_goal_transitions(attitude_target)
        next_goal_start = None
        for t in sorted(transitions):
            if t >= t_exit and not np.isnan(attitude_target[0, t]):
                next_goal_start = t
                break
        if next_goal_start is not None:
            # Find end of next goal segment (or 15% of total)
            remaining = N - next_goal_start
            ext_len = max(1, int(0.15 * remaining))
            ext_end = min(next_goal_start + ext_len, N - 1)
            extended_window = list(window) + list(range(next_goal_start, ext_end))
        else:
            extended_window = list(window)
    else:
        extended_window = list(window)

    def segment_cost(X_seg, U_seg, start_offset=0):
        total = 0.0
        for idx, k in enumerate(extended_window):
            ki = idx - (t_exit - t_enter)  # negative for main window, non-neg for extension
            if ki < 0:
                # In the main window — use PD or original trajectory
                x_k = X_seg[:, idx] if idx < X_seg.shape[1] else X_orig[:, k]
                u_k = U_seg[:, idx] if idx < U_seg.shape[1] else np.zeros(satellite.controlDim)
            else:
                # Extension — always use original tail
                x_k = X_orig[:, k]
                u_k = U_orig[:, k] if k < U_orig.shape[1] else np.zeros(satellite.controlDim)
            c = satellite.stageCost(
                k, N,
                x_k,
                u_k,
                boresight[:, k],
                attitude_target[:, k],
                B[:, k],
                cost_cfg,
            )
            total += float(c)
        return total

    cost_orig = 0.0
    cost_pd_val = 0.0
    for idx, k in enumerate(extended_window):
        in_main = idx < (t_exit - t_enter)
        if in_main:
            x_orig_k = X_orig[:, k]
            u_orig_k = U_orig[:, k] if k < U_orig.shape[1] else np.zeros(satellite.controlDim)
            x_pd_k = X_pd[:, idx]
            u_pd_k = U_pd[:, idx] if idx < U_pd.shape[1] else np.zeros(satellite.controlDim)
        else:
            # Extension window: both use original tail
            x_orig_k = X_orig[:, k]
            u_orig_k = U_orig[:, k] if k < U_orig.shape[1] else np.zeros(satellite.controlDim)
            x_pd_k = x_orig_k
            u_pd_k = u_orig_k

        def sc(x_, u_):
            return float(satellite.stageCost(
                k, N, x_, u_,
                boresight[:, k], attitude_target[:, k], B[:, k], cost_cfg,
            ))

        cost_orig += sc(x_orig_k, u_orig_k)
        cost_pd_val += sc(x_pd_k, u_pd_k)

    return cost_pd_val < cost_orig, cost_orig, cost_pd_val


# ---------------------------------------------------------------------------
# Phase 4: Keep-out check
# ---------------------------------------------------------------------------

def check_keepout(X_pd, U_pd, S, satellite, cnst_cfg, N, t_enter):
    """Return True if PD trajectory is free of sun-avoidance violations.

    Sun avoidance is constraint index 1 per alilqr.py ordering.
    """
    n_pd = X_pd.shape[1]
    nu = satellite.controlDim

    for i in range(n_pd):
        k = t_enter + i
        x_k = X_pd[:, i]
        u_k = U_pd[:, i] if i < U_pd.shape[1] else np.zeros(nu)
        S_k = S[:, k] if k < S.shape[1] else S[:, -1]

        c_k = np.asarray(satellite.constraints(k, N, x_k, u_k, S_k, cnst_cfg))
        # Index 1 = sun avoidance (constraint ordering: 0=ang_vel, 1=sun_avoid, ...)
        if len(c_k) > 1 and c_k[1] > 0.0:
            return False

    return True


# ---------------------------------------------------------------------------
# Phase 5: Blend zone + tail re-rollout
# ---------------------------------------------------------------------------

def _state_error_reduced(x_current, x_nominal, satellite):
    """MRP-based reduced state error matching the iLQR convention.

    Returns vector of size (6 + nRW): [delta_omega(3), mrp_err(3), delta_h(nRW)]
    """
    nRW = satellite.numRW
    nxr = satellite.reducedStateDim
    dx = np.zeros(nxr)

    # Angular velocity error
    dx[0:3] = x_current[0:3] - x_nominal[0:3]

    # Attitude error via MRP
    q_ref = x_nominal[3:7]
    q_cur = x_current[3:7]
    q_err = _quat_error(q_ref, q_cur)
    dx[3:6] = _quat_to_mrp(q_err)

    # RW momentum error
    for i in range(nRW):
        dx[6 + i] = x_current[7 + i] - x_nominal[7 + i]

    return dx


def substitute_and_blend(
    X,
    U,
    X_pd,
    U_pd,
    t_enter,
    t_exit,
    B_len,
    X_nominal_pre,
    U_bar,
    K_list,
    satellite,
    dist_cfg,
    B,
    S,
    R,
    V,
    rho,
    dt,
    kp_q=2.0,
    kd_w=5.0,
    rw_scale=0.0,
):
    """Substitute PD segment, blend, then re-rollout tail with iLQR gain correction.

    Parameters
    ----------
    X, U : full trajectory arrays (modified in place)
    X_pd : (nx, t_exit-t_enter+1) PD trajectory (includes state at t_exit)
    U_pd : (nu, t_exit-t_enter) PD controls
    t_enter, t_exit : spike window indices
    B_len : blend zone length (knots)
    X_nominal_pre : copy of X BEFORE substitution (used as reference for gain correction)
    U_bar : nominal iLQR controls (before gain correction) from last backward pass
    K_list : list of (nu, nxr) gain matrices indexed from 0 to N-2
    """
    N = X.shape[1]
    nx = X.shape[0]
    nu = U.shape[0]

    def _env(k):
        R_k = R[:, k] if k < R.shape[1] else R[:, -1]
        B_k = B[:, k] if k < B.shape[1] else B[:, -1]
        S_k = S[:, k] if k < S.shape[1] else S[:, -1]
        V_k = V[:, k] if k < V.shape[1] else V[:, -1]
        rho_k = int(np.round(float(rho[0, k]))) if k < rho.shape[1] else 0
        return R_k, B_k, S_k, V_k, rho_k

    # --- Substitution region [t_enter, t_exit) ---
    n_pd = t_exit - t_enter
    X[:, t_enter:t_exit] = X_pd[:, :n_pd]
    if U_pd.shape[1] >= n_pd:
        U[:, t_enter:t_exit] = U_pd[:, :n_pd]

    # The PD trajectory's last state goes into t_exit.
    # X_pd[:, n_pd] is the state at t_exit from the PD sim.
    # We now set X[:, t_exit] from the PD to anchor the blend.
    if X_pd.shape[1] > n_pd:
        X[:, t_exit] = X_pd[:, n_pd]
    # Else: X[:, t_exit] remains from before substitution (blend will overwrite it)

    # Target quaternion for blend zone PD: the spike exit state from nominal
    q_exit_target = X_nominal_pre[3:7, t_exit]

    blend_end = min(t_exit + B_len, N - 1)

    # --- Blend zone [t_exit, blend_end) ---
    for k in range(t_exit, blend_end):
        lam = float(k - t_exit) / float(B_len)  # 0 at t_exit, 1 at blend_end
        R_k, B_k, S_k, V_k, rho_k = _env(k)

        # PD contribution: target the spike exit state
        u_pd_k = _build_pd_control(X[:, k], q_exit_target, satellite, B_k, kp_q, kd_w, rw_scale=rw_scale)

        # iLQR open-loop nominal in blend zone (no gain — state is too different from spiked nominal)
        u_ilqr_k = U_bar[:, k] if k < U_bar.shape[1] else np.zeros(nu)

        u_blend = (1.0 - lam) * u_pd_k + lam * u_ilqr_k

        # Clamp to actuator limits
        n_mtq = satellite.numMTQ
        n_rw = satellite.numRW
        for i in range(n_mtq):
            u_max = float(satellite.getMTQ(i).u_max)
            u_blend[i] = float(np.clip(u_blend[i], -u_max, u_max))
        for i in range(n_rw):
            u_max = float(satellite.getRW(i).u_max)
            u_blend[n_mtq + i] = float(np.clip(u_blend[n_mtq + i], -u_max, u_max))

        U[:, k] = u_blend
        x_next = _rk4_step(satellite, X[:, k], u_blend, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k)
        if k + 1 < N:
            X[:, k + 1] = x_next

    # Tail re-rollout [blend_end, N-1) — OPEN-LOOP propagation with U_bar.
    # Must integrate to produce a feasible trajectory (X[k+1] = f(X[k], U[k])),
    # otherwise the next backward_pass linearizes around invalid states.
    # Applying feedback K·dx from the pre-substitution gains to the large dx
    # at the blend boundary blows up (state past RK4 stability), so we use
    # pure open-loop U_bar.  The safety valve catches divergence.
    for k in range(blend_end, N - 1):
        R_k, B_k, S_k, V_k, rho_k = _env(k)
        u_k = U_bar[:, k] if k < U_bar.shape[1] else np.zeros(nu)
        U[:, k] = u_k
        x_next = _rk4_step(satellite, X[:, k], u_k, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k)
        if k + 1 < N:
            X[:, k + 1] = x_next

    return X, U


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def apply_spike_removal(
    X,
    U,
    U_bar,
    K_list,
    satellite,
    plannersettings,
    pass_idx,
    R,
    V,
    B,
    S,
    rho,
    jtime,
    boresight,
    attitude_target,
    iteration,
    start_at_iter=2,
    max_intervention_iters=5,
    blend_len=30,
    goal_switch_buffer=15,
    min_consecutive=5,
    exit_fudge=2.0,
    min_prior_decrease_knots=10,
    min_spike_ratio=2.0,
    min_peak_rad=1.0,
    entry_error_max_rad=0.5,
    entry_gate_traj_min_factor=1.5,
    prior_low_max_rad=0.15,
    post_low_max_rad=0.15,
    min_post_stable_knots=10,
    post_vs_prior_ratio=0.5,
    signflip_override_post_min_max_rad=0.5,
    kp_q=2.0,
    kd_w=5.0,
    omega_max=None,
    h_max=None,
    rw_scale=-1.0,  # -1 = auto (numRW / (numMTQ + numRW))
    verbose=False,
    max_spike_knots=0,
):
    """Detect and remove trajectory spikes after an accepted iLQR forward pass.

    Parameters
    ----------
    X, U : accepted trajectory (will be modified in place if spikes found)
    U_bar : nominal controls saved BEFORE forward_pass was called
    K_list : list of gain matrices from last backward pass
    satellite, plannersettings, pass_idx : optimizer configuration
    R, V, B, S, rho : environment matrices (3/1 × N)
    jtime : time vector (for dt computation)
    boresight, attitude_target : goal specification (4 × N)
    iteration : current iLQR iteration index
    start_at_iter : don't intervene before this iteration
    max_intervention_iters : stop intervening after this many iterations
    blend_len : knots in blend zone
    goal_switch_buffer, min_consecutive, exit_fudge : detection parameters
    min_prior_decrease_knots : require this many prior-decreasing knots before spike onset
    min_spike_ratio : spike peak must be >= this multiple of entry error
    kp_q, kd_w : PD gains for substitution
    omega_max : clamp PD angular velocity to this magnitude (rad/s); prevents AL violations
    h_max : clamp PD RW momentum to this magnitude (N·m·s)
    verbose : print diagnostic info

    Returns
    -------
    X, U : (possibly modified) trajectory arrays
    substitution_occurred : bool
    """
    # Guard: only intervene in the configured iteration window
    if iteration < start_at_iter:
        if verbose:
            print(f"[SpikeRemoval] iter={iteration}: skipping (before start_at_iter={start_at_iter})")
        return X, U, False
    if iteration >= start_at_iter + max_intervention_iters:
        if verbose:
            print(f"[SpikeRemoval] iter={iteration}: skipping (past max_intervention_iters)")
        return X, U, False

    # Guard: MTQ-only configurations don't have homotopy spikes in the usual
    # sense — the trajectory naturally oscillates through hemispheres because
    # the actuators can only produce torque perpendicular to the B-field, so
    # B-field-aligned errors are uncontrollable and lead to long-period
    # oscillations.  Substituting those "spikes" with PD forces the trajectory
    # off the feasible manifold; iLQR then spends iterations recovering.
    # Empirically (2026-05-15 on 01_sat_3_0_mtq, full iter budget):
    #   with spike-removal:  PE_fin=59.7°, PE_mean=117.7°
    #   without spike-removal: PE_fin=3.7°,  PE_mean=89.3°
    # 56° PE_fin improvement, 28° PE_mean improvement.
    if satellite.numRW == 0:
        if verbose and iteration == start_at_iter:
            print(f"[SpikeRemoval] MTQ-only config (numRW=0): skipping all substitutions for this run")
        return X, U, False

    # Auto-set rw_scale based on satellite topology if user passed sentinel (-1).
    # Fraction-of-actuators: rw_scale = numRW / (numMTQ + numRW).
    #   3+0 MTQ only  → 0.0  (MTQ-only)
    #   0+3 RW only   → 1.0  (must use RW; otherwise PD produces zero torque)
    #   3+1 hybrid    → 0.25 (MTQ-dominant, RW assists)
    #   3+3 hybrid    → 0.5  (balanced)
    # Explicit override: user passes rw_scale >= 0.0 to keep current semantics.
    if rw_scale < 0.0:
        n_mtq_sat = satellite.numMTQ
        n_rw_sat = satellite.numRW
        tot = n_mtq_sat + n_rw_sat
        rw_scale = (n_rw_sat / tot) if tot > 0 else 0.0
        if verbose:
            print(f"[SpikeRemoval] iter={iteration}: auto rw_scale={rw_scale:.2f} "
                  f"(nMTQ={n_mtq_sat}, nRW={n_rw_sat})")

    pass_settings = plannersettings.passes[pass_idx]
    cost_cfg = pass_settings.cost
    cnst_cfg = plannersettings.constraints
    dist_cfg = plannersettings.disturbances

    # Compute dt from jtime
    N = X.shape[1]
    if len(jtime) >= 2:
        dt = float((jtime[1] - jtime[0]) * 36525.0 * 86400.0)
    else:
        dt = float(pass_settings.dt)

    # Compute current PE stats across the trajectory.
    theta_all = []
    for k in range(N):
        try:
            theta_all.append(_pointing_error(X, attitude_target, boresight, k))
        except Exception:
            pass
    current_mean_pe = float(np.mean(theta_all)) if theta_all else float('inf')
    current_max_pe = float(np.max(theta_all)) if theta_all else float('inf')

    # Guard: don't intervene if the trajectory is uniformly good — no knot
    # exceeds `low_pe_max_threshold` AND mean PE is not regressing.  Earlier
    # version used `mean PE < 5°` which mis-fires on trajectories that are
    # mostly converged but contain isolated late-iter spike knots (mean
    # stays low while a real spike sits in plain sight).  Switching the
    # magnitude criterion to max PE catches that case.
    if not hasattr(apply_spike_removal, '_prev_mean_pe'):
        apply_spike_removal._prev_mean_pe = float('inf')
    prev_mean_pe = apply_spike_removal._prev_mean_pe
    apply_spike_removal._prev_mean_pe = current_mean_pe

    low_pe_max_threshold_deg = 10.0
    max_pe_deg = np.degrees(current_max_pe)
    mean_pe_deg = np.degrees(current_mean_pe)
    if max_pe_deg < low_pe_max_threshold_deg and current_mean_pe <= prev_mean_pe * 1.5:
        if verbose:
            print(f"[SpikeRemoval] iter={iteration}: skipping "
                  f"(max PE {max_pe_deg:.1f}° < {low_pe_max_threshold_deg:.0f}°, "
                  f"mean {mean_pe_deg:.1f}° stable)")
        return X, U, False

    # Detect spike candidates
    candidates = detect_spikes(
        X, U, attitude_target, boresight, B, satellite, cnst_cfg,
        goal_switch_buffer=goal_switch_buffer,
        min_consecutive=min_consecutive,
        exit_fudge=exit_fudge,
        min_prior_decrease_knots=min_prior_decrease_knots,
        min_spike_ratio=min_spike_ratio,
        min_peak_rad=min_peak_rad,
        entry_error_max_rad=entry_error_max_rad,
        entry_gate_traj_min_factor=entry_gate_traj_min_factor,
        prior_low_max_rad=prior_low_max_rad,
        post_low_max_rad=post_low_max_rad,
        min_post_stable_knots=min_post_stable_knots,
        post_vs_prior_ratio=post_vs_prior_ratio,
        signflip_override_post_min_max_rad=signflip_override_post_min_max_rad,
        verbose=verbose,
    )

    if verbose:
        if not candidates:
            # Compute raw pointing error stats to explain why nothing was detected
            theta_vals = []
            for k in range(N):
                try:
                    e = _pointing_error(X, attitude_target, boresight, k)
                    theta_vals.append(e)
                except Exception:
                    theta_vals.append(float('nan'))
            theta_arr = np.array([t for t in theta_vals if not np.isnan(t)])
            print(f"[SpikeRemoval] iter={iteration}: no candidates "
                  f"(err min={np.degrees(theta_arr.min()):.1f}° "
                  f"max={np.degrees(theta_arr.max()):.1f}° "
                  f"mean={np.degrees(theta_arr.mean()):.1f}°)")
        else:
            print(f"[SpikeRemoval] iter={iteration}: {len(candidates)} candidate(s): {candidates}")

    if not candidates:
        return X, U, False

    substitution_occurred = False

    # Process candidates earliest to latest; after each substitution, the
    # trajectory changes so subsequent candidates are evaluated on updated X, U.
    for t_enter, t_exit in candidates:
        n_steps = t_exit - t_enter
        if n_steps < 2:
            continue

        # Slice environment for PD simulation window
        t_end_pd = min(t_exit + 1, N)
        B_slice = B[:, t_enter:t_end_pd]
        S_slice = S[:, t_enter:t_end_pd]
        R_slice = R[:, t_enter:t_end_pd]
        V_slice = V[:, t_enter:t_end_pd]
        rho_slice = rho[:, t_enter:t_end_pd]

        # PD target: the spike exit state, with quaternion mirrored if the
        # exit state is in the opposite hemisphere from the entry state.
        # For a single-transition spike (no return), X[t_exit] is in H2;
        # using its negative gives the H1 representation of the same
        # orientation, so PD steers smoothly in H1 without crossing
        # hemispheres.  For a pair-transition spike, X[t_exit] is already
        # back in H1, no flip needed.
        x_target = X[:, t_exit].copy()
        q_enter = X[3:7, t_enter]
        q_exit = X[3:7, t_exit]
        if float(np.dot(q_enter, q_exit)) < 0.0:
            x_target[3:7] = -x_target[3:7]
            if verbose:
                print(f"[SpikeRemoval]   ({t_enter},{t_exit}): mirroring PD target (q_enter·q_exit<0)")

        # Simulate PD segment
        X_pd, U_pd = simulate_pd_segment(
            x_start=X[:, t_enter].copy(),
            x_target=x_target,
            n_steps=n_steps,
            B_cols=B_slice,
            S_cols=S_slice,
            R_cols=R_slice,
            V_cols=V_slice,
            rho_cols=rho_slice,
            satellite=satellite,
            dist_cfg=dist_cfg,
            dt=dt,
            kp_q=kp_q,
            kd_w=kd_w,
            omega_max=omega_max,
            h_max=h_max,
            rw_scale=rw_scale,
        )

        # Validate PD sim output before using it.  At large dt (e.g., 30s)
        # the PD rollout can diverge (RK4 stiffness), producing NaN/Inf
        # quaternions.  Handing that to check_keepout or substitute_and_blend
        # crashes `satellite.constraints` on "Quaternion norm is too small".
        pd_q_norms = np.linalg.norm(X_pd[3:7, :], axis=0)
        pd_bad = (not np.all(np.isfinite(X_pd))
                  or not np.all(np.isfinite(U_pd))
                  or np.any(np.abs(pd_q_norms - 1.0) > 1e-3))
        if pd_bad:
            if verbose:
                print(f"[SpikeRemoval]   ({t_enter},{t_exit}): PD sim diverged — skipping")
            continue

        # Keep-out check
        if not check_keepout(X_pd, U_pd, S, satellite, cnst_cfg, N, t_enter):
            if verbose:
                print(f"[SpikeRemoval]   ({t_enter},{t_exit}): keep-out violation — skipping")
            continue

        if verbose:
            print(f"[SpikeRemoval]   ({t_enter},{t_exit}): substituting PD segment")

        # Save pre-substitution nominal for gain correction reference AND
        # for safety-valve rollback if substitution produces degenerate state.
        X_nominal_pre = X.copy()
        X_saved = X.copy()
        U_saved = U.copy()

        # Substitute + blend (tail rollout removed — iLQR handles that).
        X, U = substitute_and_blend(
            X, U,
            X_pd, U_pd,
            t_enter, t_exit,
            blend_len,
            X_nominal_pre,
            U_bar,
            K_list,
            satellite,
            dist_cfg,
            B, S, R, V, rho,
            dt,
            kp_q=kp_q,
            kd_w=kd_w,
            rw_scale=rw_scale,
        )

        # Safety valve: reject the substitution if it produced a degenerate
        # trajectory (NaN/Inf, quaternion norm drift, or ω far past limits).
        # Degenerate substitutions otherwise crash the next backward_pass
        # with "Quaternion norm is too small to normalize".
        q_norms = np.linalg.norm(X[3:7, :], axis=0)
        w_max = np.max(np.abs(X[0:3, :]))
        # Tolerance: |q| must be 1 ± 1e-3; ω must be within 2× the wmax
        # constraint (pipeline allows transient excursions during AL warm-up).
        wmax_safe = 2.0 * 20.0 * np.pi / 180.0   # 40°/s
        bad = (not np.all(np.isfinite(X)) or
               not np.all(np.isfinite(U)) or
               np.any(np.abs(q_norms - 1.0) > 1e-3) or
               w_max > wmax_safe)
        if bad:
            if verbose:
                reason = ("non-finite" if not np.all(np.isfinite(X)) else
                          (f"|q| drift (worst={np.max(np.abs(q_norms-1)):.3e})"
                           if np.any(np.abs(q_norms-1.0) > 1e-3) else
                           f"|ω| exceeded safe limit ({np.degrees(w_max):.1f}°/s)"))
                print(f"[SpikeRemoval]   ({t_enter},{t_exit}): safety valve — "
                      f"REVERTING substitution ({reason})")
            X = X_saved
            U = U_saved
            continue

        substitution_occurred = True
        # Only substitute one spike per call — let the backward pass re-linearize
        # before tackling any remaining spikes in the next iteration.
        break

    return X, U, substitution_occurred
