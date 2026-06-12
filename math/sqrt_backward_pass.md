# Square-Root Backward Pass

Reference: Howell, Jackson, Manchester, *ALTRO: A Fast Solver for Constrained
Trajectory Optimization*, IROS 2019, Sec. IV-A ("Square-Root Backward Pass"),
itself inspired by the square-root Kalman filter. Enabled at runtime with
`RegularizationConfig::use_sqrt_bp = true`; implemented in
`src/optimizer/backwardpass_sqrt.cpp`.

## Motivation

For augmented Lagrangian methods to converge, the penalty weights μ must grow
large, and the Gauss-Newton penalty Hessian μ ∇cᵀ∇c then dominates the
action-value Hessian. Forming and propagating these products in the dense
Riccati recursion squares the condition number: with double precision
(~16 digits), μ ≈ 1e12 leaves only ~4 digits of headroom in `P_k`. The
square-root form propagates a Cholesky-style factor `S_k` with
`P_k = S_kᵀ S_k`, so only √μ enters the recursion and the effective condition
number is halved (in the exponent).

## Recursion

All quantities are in the reduced (MRP-projected) state space, matching the
dense backward pass.

Terminal step (paper eq. 19): the terminal cost Hessian is factored, then the
terminal-knot AL penalty enters as stacked rows (state terms only — no control
exists at the terminal knot), matching the dense pass's terminal seed:

    S_N = qr([ psdSqrtFactor(G_N P_N G_Nᵀ) ; √μ_i c_x_i (active i) ])

At each step k, a square-root factor `[F_x F_u]` of the **joint** action-value
Hessian is assembled by row-stacking (cf. paper eqs. 20–21):

    [F_x F_u] = [ psdSqrtFactor([l_xx l_xu; l_ux l_uu])  ]   joint stage cost
                [ S_{k+1} [A_k  B_k]                     ]   dynamics term
                [ √μ_i [c_x_i  c_u_i]  (active i only)   ]   AL penalty rows

so that `[F_x F_u]ᵀ[F_x F_u] = [Q_xx Q_xu; Q_ux Q_uu]` (unregularized). The
penalty enters only through √μ — the μ ∇cᵀ∇c outer product is never formed.

Gains (paper eqs. 21–23): the regularized triangular factor is

    Z_uu_reg = qr([F_u; √reg · I])          (Z_uu_regᵀ Z_uu_reg = Q_uu + reg·I)
    K = −Z_uu_reg⁻¹ Z_uu_reg⁻ᵀ Q_ux,   d = −Z_uu_reg⁻¹ Z_uu_reg⁻ᵀ Q_u

with `Q_ux = F_uᵀ F_x`, `Q_x = l_x + A_kᵀ p_{k+1}`, `Q_u = l_u + B_kᵀ p_{k+1}`
(gradients are vectors and need no square-root treatment; the AL gradient
terms (λ + μc)ᵀ∇c are added to `l_x`, `l_u` exactly as in the dense pass).

Value function propagation uses the **unregularized** factor, matching the
dense pass (regularization must not inflate the cost-to-go):

    p_k     = Q_x + Kᵀ(F_uᵀ(F_u d)) + Kᵀ Q_u + Q_uxᵀ d
    ΔV     += [ dᵀQ_u,  ½‖F_u d‖² ]
    S_k     = qr(F_x + F_u K)

The last line is the paper's eq. (26):

    P_k = [I; K]ᵀ [Q_xx Q_xu; Q_ux Q_uu] [I; K] = (F_x + F_u K)ᵀ (F_x + F_u K)

which holds for **any** K — in particular the regularized one — and is PSD and
symmetric by construction, so the dense pass's explicit symmetrization is
unnecessary.

## Deviation from the paper's eqs. (27)–(29)

The paper reconstructs the joint factor from `Z_xx` and `Z_uu` via
`C = Z_xx⁻ᵀ Q_xu` and a Cholesky **downdate**
`D = √(Z_uuᵀZ_uu − CᵀC)`. This implementation instead keeps the row-stacked
joint factor `[F_x F_u]` directly, which is algebraically equivalent but:

- never inverts `Z_xx` (works when `Q_xx` is singular, e.g. zero terminal
  weight on wheel momentum);
- avoids Cholesky downdates, the numerically fragile step of the original
  scheme;
- handles a stage-cost cross term `l_ux ≠ 0` exactly;
- is consistent with SALTRO's regularized-K / unregularized-`Q_uu`
  propagation split, which the paper's `D` (built from the regularized
  `Z_uu`) is not.

The price is a QR over ~(2n_x + n_u + n_active) rows per step instead of two
n-row QRs plus downdates — negligible at SALTRO's state dimensions.

## Semantics vs. the dense pass

- Results are identical (to ~1e-10 relative) whenever the stage-cost Hessians
  are PSD; see `tests/unit/optimizer/test_backwardpass_sqrt.cpp`.
- Indefinite stage-cost Hessians (e.g. `ang_cost_func_type = 4`) are clamped
  to PSD by `psdSqrtFactor` — a real square root only exists for PSD
  matrices. The dense pass instead lets negative curvature into the
  recursion (its clamp is commented out), so results diverge on indefinite
  problems; the sqrt pass's behavior is the conservative one.
- The dense pass fails (triggering regularization escalation in iLQR) when
  `Q_uu + reg·I` is not PD. In the sqrt pass `Q_uu` is PSD by construction,
  so this failure mode disappears; the pass only reports failure on
  non-finite values or a singular `Z_uu_reg`.
