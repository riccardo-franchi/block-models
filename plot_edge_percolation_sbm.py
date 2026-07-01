import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update(
    {
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
    }
)


def compute_giant_cluster_size(c, phi=1.0, tolerance=1e-8, max_iterations=1_000):
    S_guess = np.full(c.shape[0], 0.5)
    for _ in range(max_iterations):
        S_new = 1 - np.exp(-phi * c @ S_guess)
        if np.linalg.norm(S_new - S_guess, ord=1) < tolerance:
            break
        S_guess = S_new
    return S_guess


def matrix_label(c):
    n = c.shape[0]
    col_width = max(len(f"{c[i, j]:.4g}") for i in range(n) for j in range(n))
    rows = []
    for i in range(n):
        entries = "  ".join(f"{c[i, j]:{col_width}.4g}" for j in range(n))
        bracket_l = "⎡" if i == 0 else ("⎣" if i == n - 1 else "⎢")
        bracket_r = "⎤" if i == 0 else ("⎦" if i == n - 1 else "⎥")
        rows.append(f"{bracket_l} {entries} {bracket_r}")
    label = "C ="
    padding = " " * (len(label) + 1)
    lines = [f"{label} {rows[0]}"] + [f"{padding}{row}" for row in rows[1:]]
    return "\n".join(lines)


def read_measurements(path):
    phi, s, err = [], [], []
    with open(path) as f:
        for line in f:
            p, s_val, e = map(float, line.split())
            phi.append(p)
            s.append(s_val)
            err.append(e)
    return phi, s, err


def plot_panel(ax, b, show_xlabel=True):
    c = np.array([[2.0, b], [b, 20.0]])
    node_distribution = np.array([0.5, 0.5])

    phi_values = np.linspace(0, 1, 400)
    S_values = np.array(
        [compute_giant_cluster_size(c, phi) @ node_distribution for phi in phi_values]
    )

    measured_phi, measured_s, _ = read_measurements(
        f"output/edge_percolation_sbm_b{b}.txt"
    )

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

    if show_xlabel:
        ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$S(\phi)$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
    ax.set_xlim(phi_values[0], phi_values[-1])
    ax.set_ylim(bottom=0)

    lambda_max = np.linalg.eigvalsh(c).max()
    ax.axvline(1.0 / lambda_max, color="0.4", linestyle=":", linewidth=1.0, zorder=0)
    ax.axhline(node_distribution[0], color="0.65", linestyle=":", linewidth=1.0, zorder=0)

    ax.text(
        0.97,
        0.05,
        matrix_label(c),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="0.6",
            linewidth=0.8,
            alpha=0.9,
        ),
        color="0.2",
    )


b_values = [0.1, 0.001]

fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.0), sharex=True)
for i, (ax, b) in enumerate(zip(axes, b_values)):
    plot_panel(ax, b, show_xlabel=(i == len(b_values) - 1))

# axes[0].legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper left")

plt.tight_layout()
fig.subplots_adjust(hspace=0.08)
plt.savefig("output/edge_percolation_sbm.pdf", bbox_inches="tight")
plt.show()

