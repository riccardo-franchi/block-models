import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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


a = np.array([0.5, 0.95])
node_distribution = np.array([0.9, 0.1])
avg_degrees = a / (1.0 - a)

p1 = 0.999
p2 = 1.0 - (1.0 - p1) * (
    (avg_degrees[0] * node_distribution[0]) / (avg_degrees[1] * node_distribution[1])
)

psi = np.array([[p1, 1.0 - p1], [1.0 - p2, p2]])


def g_0(x):
    return (1 - a) / (1 - a * x)


def g_1(x):
    return ((1 - a) / (1 - a * x)) ** 2


def compute_S(K, phi=1.0, tolerance=1e-8, max_iterations=1_000):
    u = np.full(K, 0.5)
    for _ in range(max_iterations):
        u_new = 1 - phi + phi * psi @ g_1(u)
        if np.linalg.norm(u_new - u, ord=1) < tolerance:
            break
        u = u_new
    return 1 - g_0(u)


phi_values = np.linspace(0, 1, 400)
S_values = np.array(
    [compute_S(a.shape[0], phi=phi) @ node_distribution for phi in phi_values]
)


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
