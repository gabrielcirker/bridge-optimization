"""
This script performs a comparative study of three lightweight beam geometries
under a unified mass constraint.  It implements the Euler–Bernoulli beam theory
for static deflection and a simplified buckling analysis, using finite
difference methods to approximate derivatives.  The three geometries are:

1. A uniform solid rectangular beam.
2. A hollow rectangular box section with uniform thickness.
3. A parabolically tapered beam whose height varies along the span.

For each geometry the script:
• Computes the second moment of area (or equivalent) and ensures total mass
  matches that of the baseline solid beam.
• Builds finite difference operators for the fourth derivative (for static
  deflection) and the second derivative (for buckling).
• Solves for the static deflection under a midspan point load.
• Solves a generalized eigenvalue problem to estimate the critical buckling
  load.
• Calculates a nondimensional strength‑to‑weight ratio by dividing the
  critical buckling load by the beam's weight.

The script prints summary statistics and plots the deflection curves.  It can
be used as a basis for an undergraduate research project or to validate
theoretical derivations in a mathematical optimization paper.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Physical and geometric parameters
E = 200e9          # Young's modulus [Pa] (e.g., steel)
rho = 7850         # density [kg/m^3]
L = 1.0            # span [m]
P = 1000.0         # applied midspan load [N]

# Discretization
N = 200            # number of interior nodes
x = np.linspace(0, L, N + 2)  # include boundary nodes at 0 and L
h = x[1] - x[0]

# Baseline solid beam dimensions (width b and height h)
b_base = 0.02      # width [m]
h_base = 0.04      # height [m]
A_base = b_base * h_base
M_target = rho * A_base * L  # total mass for all designs

# -----------------------------------------------------------------------------
# Helper functions for second moment of area and geometry definitions

def I_solid(b: float, h_: float) -> float:
    """Return the second moment of area for a solid rectangle."""
    return b * h_**3 / 12.0


def I_hollow(b: float, h_: float, t: float) -> float:
    """Return the second moment of area for a hollow rectangular section."""
    bi = b - 2 * t
    hi = h_ - 2 * t
    if bi <= 0 or hi <= 0:
        return 0.0
    I_outer = b * h_**3 / 12.0
    I_inner = bi * hi**3 / 12.0
    return I_outer - I_inner


def tapered_height(xpos: np.ndarray, span: float, h0: float, alpha: float = 0.5) -> np.ndarray:
    """Return the height profile for a parabolic taper."""
    xi = 2.0 * xpos / span - 1.0
    return h0 * (1.0 - alpha * xi**2)


def mass_tapered(b: float, h0: float, alpha: float = 0.5, num: int = 1000) -> float:
    """Return the mass of a tapered beam for a given maximum height h0."""
    xx = np.linspace(0, L, num)
    hh = tapered_height(xx, L, h0, alpha)
    area = b * hh
    return rho * np.trapz(area, xx)


def solve_h0_for_mass(b: float, mass: float, alpha: float = 0.5) -> float:
    """Solve for the maximum height h0 such that the tapered beam has a given mass."""
    low, high = 1e-4, 0.5
    for _ in range(60):
        mid = 0.5 * (low + high)
        m_mid = mass_tapered(b, mid, alpha)
        if m_mid > mass:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def solve_t_for_mass(b: float, h_: float, mass: float) -> float:
    """Solve for the wall thickness t such that the hollow beam has the target mass."""
    low, high = 1e-4, min(b, h_) / 2.0 - 1e-4
    for _ in range(60):
        mid = 0.5 * (low + high)
        bi = b - 2 * mid
        hi = h_ - 2 * mid
        if bi <= 0 or hi <= 0:
            high = mid
            continue
        area_eff = b * h_ - bi * hi
        m_mid = rho * area_eff * L
        if m_mid > mass:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


# -----------------------------------------------------------------------------
# Finite difference operators for derivatives

def fourth_derivative_matrix(n: int, dx: float) -> sp.csr_matrix:
    """Return a sparse matrix approximating the fourth derivative."""
    main = np.full(n, 6.0)
    off1 = np.full(n - 1, -4.0)
    off2 = np.full(n - 2, 1.0)
    diagonals = [main, off1, off1, off2, off2]
    offsets = [0, -1, 1, -2, 2]
    D4 = sp.diags(diagonals, offsets, shape=(n, n), format="lil") / dx**4
    return D4.tocsr()


def second_derivative_matrix(n: int, dx: float) -> sp.csr_matrix:
    """Return a sparse matrix approximating the second derivative."""
    main = -2.0 * np.ones(n)
    off = 1.0 * np.ones(n - 1)
    diagonals = [main, off, off]
    offsets = [0, -1, 1]
    D2 = sp.diags(diagonals, offsets, shape=(n, n), format="csr") / dx**2
    return D2


# -----------------------------------------------------------------------------
# Solvers for static deflection and buckling

def solve_deflection(EI: np.ndarray, load_type: str = "point", x_load: float = L / 2, p_load: float = P) -> np.ndarray:
    """
    Solve the static deflection problem EI(x) y'''' = q(x) with simply supported
    boundary conditions.  The load can be a point load at x_load or uniform.
    Returns the full deflection array including boundary nodes.
    """
    n_internal = N
    D4 = fourth_derivative_matrix(n_internal, h)

    # Build diagonal matrix of EI values
    if np.isscalar(EI):
        EI_vec = EI * np.ones(n_internal)
    else:
        EI_vec = EI
    EI_mat = sp.diags(EI_vec, 0, shape=(n_internal, n_internal))
    K = EI_mat.dot(D4)

    # Load vector q
    q_vec = np.zeros(n_internal)
    if load_type == "point":
        idx = np.argmin(np.abs(x[1:-1] - x_load))
        q_vec[idx] = -p_load / h  # approximate Dirac delta
    elif load_type == "uniform":
        q_vec[:] = -p_load / L

    # Solve K y_internal = q_vec
    y_internal = spla.spsolve(K, q_vec)
    y_full = np.zeros(n_internal + 2)
    y_full[1:-1] = y_internal
    return y_full


def solve_buckling(EI: np.ndarray) -> float:
    """
    Solve the buckling eigenvalue problem -EI(x) y'' = P y with Dirichlet
    boundary conditions, returning the smallest positive eigenvalue.
    """
    n_internal = N
    D2 = second_derivative_matrix(n_internal, h)
    if np.isscalar(EI):
        EI_vec = EI * np.ones(n_internal)
    else:
        EI_vec = EI
    EI_mat = sp.diags(EI_vec, 0, shape=(n_internal, n_internal))
    A = -EI_mat.dot(D2)
    vals, _ = spla.eigs(A, k=3, which='SM')
    vals = np.real(vals)
    vals_sorted = np.sort(vals)
    return vals_sorted[0]


# -----------------------------------------------------------------------------
# Geometry builders and comparison logic

def geometry_solid():
    EI = E * I_solid(b_base, h_base)
    mass = rho * b_base * h_base * L
    return EI, mass


def geometry_hollow():
    t = solve_t_for_mass(b_base, h_base, M_target)
    EI = E * I_hollow(b_base, h_base, t)
    return EI, M_target


def geometry_tapered(alpha: float = 0.5):
    h0 = solve_h0_for_mass(b_base, M_target, alpha)
    # Evaluate h(x) at interior nodes only for EI(x)
    h_profile = tapered_height(x[1:-1], L, h0, alpha)
    EI_values = E * I_solid(b_base, h_profile)
    return EI_values, M_target


def compare_geometries():
    # Solid
    EI_solid, mass_solid = geometry_solid()
    y_solid = solve_deflection(EI_solid)
    Pcr_solid = solve_buckling(EI_solid)
    SWR_solid = Pcr_solid / (mass_solid * 9.81)
    max_def_solid = np.max(np.abs(y_solid))

    # Hollow
    EI_hollow, mass_hollow = geometry_hollow()
    y_hollow = solve_deflection(EI_hollow)
    Pcr_hollow = solve_buckling(EI_hollow)
    SWR_hollow = Pcr_hollow / (mass_hollow * 9.81)
    max_def_hollow = np.max(np.abs(y_hollow))

    # Tapered
    EI_taper, mass_taper = geometry_tapered(alpha=0.5)
    y_taper = solve_deflection(EI_taper)
    Pcr_taper = solve_buckling(EI_taper)
    SWR_taper = Pcr_taper / (mass_taper * 9.81)
    max_def_taper = np.max(np.abs(y_taper))

    return {
        "solid": (SWR_solid, max_def_solid, Pcr_solid),
        "hollow": (SWR_hollow, max_def_hollow, Pcr_hollow),
        "tapered": (SWR_taper, max_def_taper, Pcr_taper),
        "deflections": (y_solid, y_hollow, y_taper)
    }


def main() -> None:
    results = compare_geometries()
    # Unpack results
    SWR_solid, max_def_solid, Pcr_solid = results["solid"]
    SWR_hollow, max_def_hollow, Pcr_hollow = results["hollow"]
    SWR_taper, max_def_taper, Pcr_taper = results["tapered"]
    y_solid, y_hollow, y_taper = results["deflections"]

    # Print summary
    print(
        f"Solid: SWR = {SWR_solid:.2f}, Max deflection = {max_def_solid:.6e} m, Pcr = {Pcr_solid:.2f} N"
    )
    print(
        f"Hollow: SWR = {SWR_hollow:.2f}, Max deflection = {max_def_hollow:.6e} m, Pcr = {Pcr_hollow:.2f} N"
    )
    print(
        f"Tapered: SWR = {SWR_taper:.2f}, Max deflection = {max_def_taper:.6e} m, Pcr = {Pcr_taper:.2f} N"
    )

    # Plot deflection curves
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_solid, label="Solid")
    plt.plot(x, y_hollow, label="Hollow")
    plt.plot(x, y_taper, label="Tapered")
    plt.xlabel("x (m)")
    plt.ylabel("Deflection (m)")
    plt.title("Static Deflection under Midspan Point Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()