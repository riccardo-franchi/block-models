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


def compute_S(K, k_max, tolerance=1e-8, max_iterations=1_000):

    u = np.full(K, 0.5)  # Initial guess

    one = np.ones(K)

    f_1_at_1 = f_1(one, k_max)

    for iteration in range(max_iterations):
        u_new = 1 - psi @ (f_1_at_1 - f_1(u, k_max))
        if np.linalg.norm(u_new - u, ord=1) < tolerance:
            break
        u = u_new

    S = f_0(one, k_max) - f_0(u, k_max)

    return S


a = np.array([0.5, 0.95])

avg_degrees = a / (1.0 - a)

p1 = 0.999
p2 = 1.0 - (1.0 - p1) * (avg_degrees[0] / avg_degrees[1])

psi = np.array([[p1, 1.0 - p1], [1.0 - p2, p2]])


def f_0(z, k_max):
    return (1 - a) * (1 - (a * z) ** (k_max + 1)) / (1 - a * z)


def f_1(z, k_max):
    return ((1 - a) / (1 - a * z)) ** 2 * (
        1 - (a * z) ** (k_max + 1) - (1 - a * z) * (k_max + 1) * (a * z) ** k_max
    )


k_values = np.arange(0, 200)
S_values = []
occupation_values = []

for k in k_values:
    S_vec = compute_S(a.shape[0], k_max=k)
    S_values.append(np.average(S_vec))
    occupation_values.append(np.mean(f_0(np.ones(a.shape[0]), k_max=k)))

S_values = np.array(S_values)
occupation_values = np.array(occupation_values)


measured_avg_phi, measured_s = [], []
with open("s_k.txt") as f:
    for line in f:
        _k, phi, s = map(float, line.split())
        measured_avg_phi.append(phi)
        measured_s.append(s)

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(occupation_values, S_values, color="#2166ac", label="Analytical", zorder=1)
ax.scatter(
    measured_avg_phi,
    measured_s,
    marker="o",
    s=1,
    color="#d6604d",
    label="Simulation",
    zorder=2,
)

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$S(\phi)$")
# ax.set_title("Giant component size vs. edge occupation probability")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)

ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")

plt.tight_layout()
plt.savefig("giant_cluster_size.png", dpi=300, bbox_inches="tight")
plt.show()
