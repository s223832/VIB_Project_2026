"""
Test of the CF cost function on the simply supported beam, using a
local reduction in E to represent damage.
Test of the CF cost function on the simply supported beam - damage 
localization and quantification using a vector theta (one free parameter
per element), optimized with Nelder-Mead.

Build the beam model at a KNOWN "true" damage severity theta_true
(E reduced only in one element, representing e.g. a crack or
section loss) and treat its output as synthetic "measured" data

Wrap CF so it rebuilds the model at any trial theta and scores it
against the synthetic measurement.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions.model_update.MAC import MAC

from classes.VIBframe import VIBframe   
from functions.model_update.CF import CF

# 1. Simply supported beam geometry 
L = 10.0
n_elem = 10
n_node = n_elem + 1
n_prop = n_elem

# Build X and C (each sub element given a seperat propno)
X = np.array([[x, 0.0, 0.0] for x in np.linspace(0, L, n_node)])
C = np.array([[i + 1, i + 2, i + 1] for i in range(n_elem)])

E0 = 210e9
rho = 7850.0
nu = 0.3
G = E0 / (2 * (1 + nu))
b = 0.1
A = b**2
Iz = b**4 / 12
Iy = b**4 / 12
J = 2 * Iz

# In-plane bending boundary conditions
bound_list = []
for node in range(1, n_node + 1):
    bound_list.append((node, 3, 0.0))   # uz
    bound_list.append((node, 4, 0.0))   # rx
    bound_list.append((node, 5, 0.0))   # ry
bound_list.append((1, 1, 0.0))          # pin: ux
bound_list.append((1, 2, 0.0))          # pin: uy
bound_list.append((n_node, 2, 0.0))     # roller: uy
bound = np.array(bound_list)

spring_support = np.array([])  # no springs
N_modes = 5

# 2. Generate synthetic "measured" data at a KNOWN true damage
def mprop_for(theta_vector):
    """propno k gets E scaled by theta_vector[k-1]."""
    return {
        k: {'E': E0 * theta_vector[k - 1], 'A': A, 'rho': rho,
            'Iy': Iy, 'Iz': Iz, 'J': J, 'G': G, 'type': 'beam'}
        for k in range(1, n_prop + 1)
    }

damaged_element = 4
theta_true = 0.6   # 40% stiffness loss in the damaged element

theta_true_vector = np.ones(n_prop)
theta_true_vector[damaged_element] = 0.6
 
model_true = VIBframe(X, C, mprop_for(theta_true_vector), bound, spring_support)
omega_meas = model_true.omega[:N_modes]
U_meas = model_true.U[:, :N_modes]

# 3. Cost function 
w_lambda = 1.0
w_MAC = 1.0

def scalar_CF(theta_vector):
    model = VIBframe(X, C, mprop_for(theta_vector), bound, spring_support)
    omega_num = model.omega[:N_modes]
    U_num = model.U[:, :N_modes]
    return CF(omega_num, omega_meas, U_num, U_meas, w_lambda, w_MAC)

# 4. Sanity check: cost should be (numerically) zero at the true theta_vector
cost_at_truth = scalar_CF(theta_true_vector)
print(f"Cost at true theta_vector: {cost_at_truth:.3e}  (should be ~0)")
print()
 
# 5. Optimize with Nelder-Mead, starting from assumption of no damage (i.e. full E modulus)
theta0 = np.ones(n_prop)
omega_history = []
mac_history = []
cost_history = []
     
def record_iteration(theta_vector):
    model = VIBframe(X, C, mprop_for(theta_vector), bound, spring_support)
    omega_num = model.omega[:N_modes]
    U_num = model.U[:, :N_modes]
 
    mac_vals = np.array([MAC(U_num[:, i], U_meas[:, i]) for i in range(N_modes)])
    cost_val = CF(omega_num, omega_meas, U_num, U_meas, w_lambda, w_MAC)
 
    omega_history.append(omega_num.copy())
    mac_history.append(mac_vals.copy())
    cost_history.append(cost_val)
 
# record the starting guess too, so the convergence plots include iteration 0
record_iteration(theta0)
res_nm = minimize(
    scalar_CF, theta0,
    method='Nelder-Mead',
    bounds=[(0.2, 2.0)] * n_prop,
    options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 5000},
    callback=record_iteration,
)
theta_recovered = res_nm.x
 
print("Recovered vs true theta per element:")
print(f"{'Element':>8} {'True theta':>12} {'Recovered':>12} {'Diff':>10}")
for i in range(n_prop):
    diff = theta_recovered[i] - theta_true_vector[i]
    marker = "  <-- damaged" if i == damaged_element else ""
    print(f"{i+1:>8} {theta_true_vector[i]:>12.4f} {theta_recovered[i]:>12.4f} {diff:>10.4f}{marker}")
 
print()
print(f"Optimizer converged: {res_nm.success}, final cost = {res_nm.fun:.3e}")


# 7. Plot convergence of natural frequencies, and MAC values, per mode   
omega_history_arr = np.array(omega_history)
lambda_history_arr = omega_history_arr**2
mac_history_arr = np.array(mac_history)
cost_history_arr = np.array(cost_history)       
iterations = np.arange(lambda_history_arr.shape[0])
 
# Relative frequency error per mode per iteration:
lambda_meas = omega_meas**2
lambda_rel_error = (lambda_history_arr - lambda_meas) / lambda_meas
mac_error = 1.0 - mac_history_arr
 
# --- Plot 1: relative frequency error convergence, one line per mode ---
plt.figure()
for i in range(N_modes):
    plt.plot(iterations, lambda_rel_error[:, i], marker='o',
            markersize=3, label=f'Mode {i+1}')
plt.axhline(0.0, color='gray', linestyle=':', linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Relative error (lambda_num - lambda_meas) / lambda_meas')
plt.title('Convergence of relative error of eigenvalues')
plt.legend(fontsize=8)
plt.savefig('CF_lambda_error_convergence.png', dpi=150)
print("Saved plot to CF_lambda_error_convergence.png")
 
# --- Plot 2: MAC error convergence, one line per mode ---
plt.figure()
for i in range(N_modes):
    plt.plot(iterations, mac_error[:, i], marker='o',
            markersize=3, label=f'Mode {i+1}')
plt.axhline(0.0, color='gray', linestyle=':', linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('MAC error (1 - MAC)')
plt.title('Convergence of MAC error during optimization')
plt.legend(fontsize=8)
plt.savefig('CF_MAC_error_convergence.png', dpi=150)
print("Saved plot to CF_MAC_error_convergence.png")

# --- Plot 3: cost function (CF) convergence ---
plt.figure()
plt.plot(iterations, cost_history_arr, marker='o', markersize=3, color='tab:red')
plt.xlabel('Iteration')
plt.ylabel('Cost function value, CF(theta)')
plt.title('Convergence of the cost function during optimization')
plt.savefig('CF_cost_convergence.png', dpi=150)
print("Saved plot to CF_cost_convergence.png")
 

if np.all(cost_history_arr > 0):
    plt.figure()
    plt.semilogy(iterations, cost_history_arr, marker='o', markersize=3, color='tab:red')
    plt.xlabel('Iteration')
    plt.ylabel('Cost function value, CF(theta) [log scale]')
    plt.title('Convergence of the cost function during optimization (log scale)')
    plt.savefig('CF_cost_convergence_log.png', dpi=150)
    print("Saved plot to CF_cost_convergence_log.png")
 