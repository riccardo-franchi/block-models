import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import special

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
})

phi_values = np.linspace(0, 1, 400)


def read_measurements(path):
    phi, s, err = [], [], []
    with open(path) as f:
        for line in f:
            p, s_val, e = map(float, line.split())
            phi.append(p)
            s.append(s_val)
            err.append(e)
    return phi, s, err


def style_axes(ax, show_xlabel=True):
    if show_xlabel:
        ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$S(\phi)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
    ax.set_xlim(phi_values[0], phi_values[-1])
    ax.set_ylim(bottom=0)


def draw_panel(ax, S_values, measured_path, show_xlabel=True):
    measured_phi, measured_s, _ = read_measurements(measured_path)
    ax.plot(phi_values, S_values, color="#2166ac", label="Analytical", zorder=1)
    ax.scatter(
        measured_phi,
        measured_s,
        s=20,
        color="#d6604d",
        label="Simulation",
        zorder=2,
        linewidths=0,
    )
    style_axes(ax, show_xlabel=show_xlabel)


# Panel 1: two geometric groups, parameters a = (0.4, 0.8)

a_geo = np.array([0.4, 0.8])
node_distribution_geo = np.array([0.5, 0.5])


def g0_geometric(u, a):
    return (1 - a) / (1 - a * u)


def g1_geometric(u, a):
    return ((1 - a) / (1 - a * u)) ** 2


avg_degrees_geo = a_geo / (1.0 - a_geo)

p1 = 0.999
p2_geo = 1.0 - (1.0 - p1) * (
    (avg_degrees_geo[0] * node_distribution_geo[0])
    / (avg_degrees_geo[1] * node_distribution_geo[1])
)
psi_geo = np.array([[p1, 1.0 - p1], [1.0 - p2_geo, p2_geo]])


def compute_S_geometric(phi=1.0, tolerance=1e-8, max_iterations=1_000):
    u = np.full(2, 0.5)
    for _ in range(max_iterations):
        g1_vals = g1_geometric(u, a_geo)
        u_new = 1 - phi + phi * (psi_geo @ g1_vals)
        if np.linalg.norm(u_new - u, ord=1) < tolerance:
            break
        u = u_new
    S = 1 - g0_geometric(u, a_geo)
    return S


S_values_geo = np.array([
    compute_S_geometric(phi) @ node_distribution_geo for phi in phi_values
])


# Panel 2: group 0 geometric (a = 0.5), group 1 power law (alpha = 2.5)


a = 0.5
alpha = 2.5
N = 100_000  # truncation for power law series

node_distribution = np.array([0.9, 0.1])


def g0_geo(u):
    return (1 - a) / (1 - a * u)


def g1_geo(u):
    return ((1 - a) / (1 - a * u)) ** 2


avg_degree_geo = a / (1.0 - a)


def precompute_p(alpha, N):
    k_values = np.arange(1, N + 1)
    p_values = k_values ** (-alpha) / special.zeta(alpha, 1.0)
    return k_values, p_values


k_pl, p_pl = precompute_p(alpha, N)
pk_pl = p_pl * k_pl
km1_pl = k_pl - 1
avg_degree_pl = np.sum(pk_pl)


def _powers(u):
    # u^0, u^1, ..., u^(N-1) via cumulative product, which is much cheaper
    # than np.power(u, km1_pl) since it only does N-1 multiplications
    # instead of N general (float base, float exponent) pow() calls.
    u = np.asarray(u)
    powers = np.broadcast_to(u[..., None], u.shape + (N,)).copy()
    powers[..., 0] = 1.0
    np.cumprod(powers, axis=-1, out=powers)
    return powers


def g0_pl(u):
    return (_powers(u) * np.asarray(u)[..., None]) @ p_pl


def g1_pl(u):
    return _powers(u) @ pk_pl / avg_degree_pl


avg_degrees = np.array([avg_degree_geo, avg_degree_pl])

p2 = 1.0 - (1.0 - p1) * (
    (avg_degrees[0] * node_distribution[0]) / (avg_degrees[1] * node_distribution[1])
)
psi = np.array([[p1, 1.0 - p1], [1.0 - p2, p2]])


def compute_S(phi, tolerance=1e-8, max_iterations=1_000):
    # Solve the fixed point for every phi at once (vectorized) instead of
    # looping over phi in Python, since g0_pl/g1_pl dominate the cost.
    # Points near the percolation threshold converge much slower than the
    # rest, so only keep iterating on the columns that haven't converged
    # yet instead of forcing every phi through max_iterations.
    u = np.full((2, phi.shape[0]), 0.5)
    active = np.ones(phi.shape[0], dtype=bool)
    for _ in range(max_iterations):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        u_active = u[:, idx]
        phi_active = phi[idx]
        g1_vals = np.stack([g1_geo(u_active[0]), g1_pl(u_active[1])])
        u_new = (1 - phi_active) + phi_active * (psi @ g1_vals)
        converged = np.linalg.norm(u_new - u_active, ord=1, axis=0) < tolerance
        u[:, idx] = u_new
        active[idx] = ~converged
    S = np.stack([1 - g0_geo(u[0]), 1 - g0_pl(u[1])])
    return S


S_values_mixed = node_distribution @ compute_S(phi_values)


fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.0), sharex=True)

draw_panel(
    axes[0],
    S_values_geo,
    "output/edge_percolation_dc_sbm_geometric.txt",
    show_xlabel=False,
)
draw_panel(
    axes[1],
    S_values_mixed,
    "output/edge_percolation_dc_sbm_mixed.txt",
    show_xlabel=True,
)

# axes[0].legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper left")

plt.tight_layout()
fig.subplots_adjust(hspace=0.08)
plt.savefig("output/edge_percolation_dc_sbm.pdf", bbox_inches="tight")
plt.show()
