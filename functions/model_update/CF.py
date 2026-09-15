
import numpy as np
from functions.model_update.MAC import MAC
from functions.model_update.relative_diff import relative_diff


def CF(omega_num, omega_meas, U_num, U_meas, w_lambda, w_MAC):
    """
    Scalar cost function for model updating, combining an eigenvalue
    (natural frequency) term and a mode shape (MAC) term.
 
    Parameters
    ----------
    omega_num : array_like, shape (N_modes,)
        Natural frequencies from the FE model [rad/s]
    omega_meas : array_like, shape (N_modes,)
        Natural frequencies identified from OMA [rad/s], same mode order
        as omega_num
    U_num : array_like, shape (ndof, N_modes)
        FE mode shapes, one column per mode
    U_meas : array_like, shape (ndof, N_modes)
        Measured/OMA mode shapes, one column per mode, same mode order
        as U_num
    w_lambda : float
        Weight applied to the eigenvalue contribution
    w_MAC : float
        Weight applied to the MAC contribution
 
    Returns
    -------
    float
        Weighted scalar cost value (0 = perfect match on both criteria)
    """

    omega_num = np.asarray(omega_num).flatten()
    omega_meas = np.asarray(omega_meas).flatten()
 
    if omega_num.shape != omega_meas.shape:
        raise ValueError(
            f"omega_num and omega_meas must have the same length, "
            f"got {omega_num.shape} and {omega_meas.shape}"
        )
 
    U_num = np.asarray(U_num)
    U_meas = np.asarray(U_meas)
 
    if U_num.shape != U_meas.shape:
        raise ValueError(
            f"U_num and U_meas must have the same shape, "
            f"got {U_num.shape} and {U_meas.shape}"
        )
 
    N_modes = omega_num.shape[0]
 
    if U_num.shape[1] != N_modes:
        raise ValueError(
            f"Number of mode shape columns ({U_num.shape[1]}) must match "
            f"the number of frequencies ({N_modes})"
        )
 
    # --- Eigenvalue (frequency) contribution ---
    lambda_res = relative_diff(omega_num, omega_meas)         
    con_lambda = np.sum(lambda_res ** 2)

    # --- Mode shape (MAC) contribution ---
    mac_vals = np.array([MAC(U_num[:, i], U_meas[:, i]) for i in range(N_modes)])
    con_MAC = np.sum((1.0 - mac_vals) ** 2)    
 
    return w_lambda * con_lambda + w_MAC * con_MAC

