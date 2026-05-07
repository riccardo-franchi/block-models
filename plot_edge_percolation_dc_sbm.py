import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import special

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 11,
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
    }
)

# Group 0: geometric with parameter p = 1 - a  (P(k) = (1-a)*a^k, k=0,1,2,...)
# Group 1: power law (Zeta) with exponent alpha
a = 0.5
alpha = 2.5
N = 100_000  # truncation for power law series

node_distribution = np.array([0.9, 0.1])

# --- Group 0: geometric generating functions ---


def g0_geo(u):
    return (1 - a) / (1 - a * u)


def g1_geo(u):
    return ((1 - a) / (1 - a * u)) ** 2


avg_degree_geo = a / (1.0 - a)

# --- Group 1: power law generating functions ---


def precompute_p(alpha, N):
    k_values = np.arange(1, N + 1)
    p_values = k_values ** (-alpha) / special.zeta(alpha, 1.0)
    return k_values, p_values


k_pl, p_pl = precompute_p(alpha, N)
avg_degree_pl = np.sum(k_pl * p_pl)


def g0_pl(u):
    return np.sum(p_pl * u**k_pl)


def g1_pl(u):
    return np.sum(p_pl * k_pl * u ** (k_pl - 1)) / avg_degree_pl


# ---  ---

avg_degrees = np.array([avg_degree_geo, avg_degree_pl])

p1 = 0.999
p2 = 1.0 - (1.0 - p1) * (
    (avg_degrees[0] * node_distribution[0]) / (avg_degrees[1] * node_distribution[1])
)

psi = np.array([[p1, 1.0 - p1], [1.0 - p2, p2]])


def compute_S(phi=1.0, tolerance=1e-8, max_iterations=1_000):
    u = np.full(2, 0.5)
    for _ in range(max_iterations):
        g1_vals = np.array([g1_geo(u[0]), g1_pl(u[1])])
        u_new = 1 - phi + phi * (psi @ g1_vals)
        if np.linalg.norm(u_new - u, ord=1) < tolerance:
            break
        u = u_new
    S = np.array([1 - g0_geo(u[0]), 1 - g0_pl(u[1])])
    return S


phi_values = np.linspace(0, 1, 400)
S_values = np.array([compute_S(phi) @ node_distribution for phi in phi_values])


measured_phi, measured_s, measured_err = [], [], []
with open("output/edge_percolation_dc_sbm.txt") as f:
    for line in f:
        p, s_val, err = map(float, line.split())
        measured_phi.append(p)
        measured_s.append(s_val)
        measured_err.append(err)

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(phi_values, S_values, color="#2166ac", label="Analytical", zorder=1)
ax.errorbar(
    measured_phi,
    measured_s,
    yerr=measured_err,
    fmt="o",
    markersize=1,
    color="#d6604d",
    label="Simulation",
    zorder=2,
    capsize=2,
    linewidth=0.8,
    elinewidth=0.8,
)

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$S(\phi)$")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.set_xlim(phi_values[0], phi_values[-1])
ax.set_ylim(bottom=0)

ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")

plt.tight_layout()
plt.savefig("output/edge_percolation_dc_sbm.svg", bbox_inches="tight")
plt.show()
