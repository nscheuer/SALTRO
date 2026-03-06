import numpy as np
import saltro_py


def discretize_trajectory(boresight_sparse, qgoal_sparse, time_sparse, dt):
    SEC_PER_CENTURY = 36525.0 * 86400.0
    t_start = time_sparse[0, 0]
    t_end = time_sparse[0, -1]
    dt_centuries = dt / SEC_PER_CENTURY
    
    N = int(np.ceil((t_end - t_start) / dt_centuries)) + 1
    time_dense = np.zeros(N)
    for k in range(N):
        time_dense[k] = t_start + k * dt_centuries
    
    boresight_dense = np.zeros((3, N))
    qgoal_dense = np.zeros((4, N))
    
    for i in range(3):
        boresight_dense[i, :] = np.interp(time_dense, time_sparse[0, :], boresight_sparse[i, :])
    
    for i in range(4):
        qgoal_dense[i, :] = np.interp(time_dense, time_sparse[0, :], qgoal_sparse[i, :])
    
    for k in range(N):
        q = qgoal_dense[:, k]
        qgoal_dense[:, k] = q / np.linalg.norm(q)
    
    return boresight_dense, qgoal_dense, time_dense


def ilqr(satellite, settings, X, U, jtime, R, V, B, S, rho, boresight, qgoal):
    reg_cfg = settings.passes[0].reg
    ilqr_cfg = settings.passes[0].ilqr
    
    reg = reg_cfg.reg_init
    
    for iteration in range(ilqr_cfg.max_iters):
        U_trim = U[:, :X.shape[1] - 1]
        
        while reg <= reg_cfg.reg_max:
            ok_bp, K_arr, d_arr, deltaV = saltro_py.backward_pass(
                satellite, X, U_trim, R, V, B, S, rho, boresight, qgoal, settings, reg
            )
            
            if not ok_bp:
                reg *= reg_cfg.reg_scale
                continue
            
            K_list = [K_arr[k] for k in range(K_arr.shape[0])]
            d_list = [d_arr[:, k] for k in range(d_arr.shape[1])]
            
            J_prev = satellite.totalCost(X, U_trim, B, boresight, qgoal, settings.passes[0].cost)
            
            ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(
                satellite, X, U, K_list, d_list, deltaV, B, R, V, S, rho,
                boresight, qgoal, settings, jtime, J_prev
            )
            
            if not ok_fp:
                reg *= reg_cfg.reg_scale
                continue
            
            X = X_new
            U = U_new
            
            delta_J = abs(J_prev - J_new)
            if delta_J <= ilqr_cfg.cost_tol:
                return X, U, "converged"
            
            reg = max(reg_cfg.reg_init, reg / reg_cfg.reg_scale)
            break
        
        if reg > reg_cfg.reg_max:
            return X, U, "reg_exceeded"
    
    return X, U, "max_iters"


def trajOpt(satellite, settings, x0, boresight_sparse, qgoal_sparse, time_sparse, R, V, B, S, rho):
    dt = settings.passes[0].dt
    
    boresight, qgoal, jtime = discretize_trajectory(boresight_sparse, qgoal_sparse, time_sparse, dt)
    
    ok, X, U = saltro_py.warm_start(
        settings, satellite, x0, jtime, qgoal, boresight, R, V, B, S, rho
    )
    
    if not ok:
        raise RuntimeError("warm_start failed")
    
    X_opt, U_opt, reason = ilqr(satellite, settings, X, U, jtime, R, V, B, S, rho, boresight, qgoal)
    
    return X_opt, U_opt, reason, jtime, boresight, qgoal
