"""
Test case: simply supported beam (Euler-Bernoulli bending)
 
Compares the natural frequencies computed by VIBframe against the
analytical solution for a simply supported prismatic beam:
 
    omega_n = (n * pi / L)^2 * sqrt(E * I / (rho * A))
 
The model is restrained to pure bending in the x-y plane:
  - uz, rx (torsion), ry are fixed at EVERY node (removes out-of-plane
    bending, torsion, and the second bending plane)
  - ux, uy fixed at node 1 (pin)
  - uy fixed at the last node (roller)
 
"""

# Import functions
import numpy as np
from math import pi, sqrt

from functions.plot.plotmodeshapes import plotmodeshapes
from functions.data.basestore import basestore
from functions.data.output import output

# Import classes
from classes.VIBframe import VIBframe
from classes.VIBdata import VIBdata

# 1. Beam geometry and mesh
L = 10.0          # beam length [m]
n_elem = 10       # number of elements
n_node = n_elem + 1
 
X = np.array([[x, 0.0, 0.0] for x in np.linspace(0, L, n_node)])
C = np.array([[i + 1, i + 2, 1] for i in range(n_elem)])  # node1, node2, propno
 
# 2. Material / section properties (square steel section, 100x100 mm)
E = 210e9          # Young's modulus [Pa]
rho = 7850.0       # density [kg/m^3]
nu = 0.3
G = E / (2 * (1 + nu))

b = 0.1            # section width/height [m]
A = b**2
Iz = b**4 / 12     # bending about z (governs x-y plane bending, restrained here)
Iy = b**4 / 12
J = 2 * Iz         # rough torsional constant for a square section (not used, torsion is restrained)
 
mprop = {1: {'E': E, 'A': A, 'rho': rho, 'Iy': Iy, 'Iz': Iz, 'J': J, 'G': G, 'type': 'beam'}}

# 3. Boundary conditions
bound_list = []

# Restrain out-of-plane motion at every node: uz, rx (torsion), ry
for node in range(1, n_node + 1):
    bound_list.append((node, 3, 0.0))
    bound_list.append((node, 4, 0.0))
    bound_list.append((node, 5, 0.0))

# Pin support at node 1: restrain ux, uy
bound_list.append((1, 1, 0.0))
bound_list.append((1, 2, 0.0))
 
# Roller support at last node: restrain uy
bound_list.append((n_node, 2, 0.0))
bound = np.array(bound_list)
spring_support = np.array([])  # no springs

# 4. Run the analysis
VIB = VIBframe(X, C, mprop, bound, spring_support)

output(VIB, 'VIB_results.txt')
basestore(VIB,name='VIB_DataBase.db')
data_vib = VIBdata(name='VIB_DataBase.db')

omega_numerical = VIB.omega

# 5. Analytical solution for comparison
n_modes = min(6, len(omega_numerical))
omega_analytical = np.array([
    (n * pi / L) ** 2 * sqrt(E * Iz / (rho * A)) for n in range(1, n_modes + 1)
])

# 6. Print comparison
print(f"{'Mode':>5} {'Numerical [rad/s]':>20} {'Analytical [rad/s]':>20} {'Error [%]':>12}")
print("-" * 60)
for i in range(n_modes):
    num = omega_numerical[i]
    ana = omega_analytical[i]
    err = 100 * (num - ana) / ana
    print(f"{i+1:>5} {num:>20.4f} {ana:>20.4f} {err:>12.3f}")

# 7. Plot mode shapes
for i in range(0, 5):
    plotmodeshapes(data_vib, mode=i, scale=0.5)
