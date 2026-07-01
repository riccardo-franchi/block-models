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


def compute_pi_via_fft(g0, g1, psi, max_size, n_points=None, tol=1e-13, max_iter=1000):
    K = len(g0)
    if n_points is None:
        n_points = max(4 * (max_size + 1), 256)

    h0_on_circle = np.zeros((K, n_points), dtype=complex)

    for n in range(n_points):
        z = np.exp(2j * np.pi * n / n_points)
        h1 = np.zeros(K, dtype=complex)
        for _ in range(max_iter):
            arg = psi @ h1
            h1_new = z * np.array([g1[r](arg[r]) for r in range(K)])
            if np.max(np.abs(h1_new - h1)) < tol:
                h1 = h1_new
                break
            h1 = h1_new
        arg = psi @ h1
        h0_on_circle[:, n] = z * np.array([g0[r](arg[r]) for r in range(K)])

    # pi^r_s = (1/N) sum_n h0^r(z_n) e^{-2*pi*i*n*s/N} = fft(h0_on_circle)[s] / N
    coeffs = (np.fft.fft(h0_on_circle, axis=1) / n_points).real
    return coeffs[:, : max_size + 1]


a = np.array([0.2, 0.4])
p1 = 0.8
avg_deg = a / (1.0 - a)
p2 = 1.0 - (1.0 - p1) * (avg_deg[0] / avg_deg[1])
psi = np.array([[p1, 1.0 - p1], [1.0 - p2, p2]])

g0 = [lambda z, ai=ai: (1 - ai) / (1 - ai * z) for ai in a]
g1 = [lambda z, ai=ai: ((1 - ai) / (1 - ai * z)) ** 2 for ai in a]

max_size = 50
pi_r = compute_pi_via_fft(g0, g1, psi, max_size)
pi_fft = pi_r.mean(axis=0)

sizes, analytical, simulated = [], [], []
with open("output/small_components.txt") as f:
    for line in f:
        s, ana, sim = line.split()
        sizes.append(int(s))
        analytical.append(float(ana))
        simulated.append(float(sim))

sizes = np.array(sizes)
analytical = np.array(analytical)
simulated = np.array(simulated)
pi_fft_plot = pi_fft[sizes]


fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(sizes, analytical, color="#2166ac", label="Analytical (polynomial)", zorder=1)
# ax.plot(
#     sizes,
#     pi_fft_plot,
#     color="#1a9850",
#     linestyle=":",
#     label="Numerical (Cauchy / FFT)",
#     zorder=2,
# )
ax.scatter(
    sizes,
    simulated,
    s=20,
    color="#d6604d",
    label="Simulation",
    zorder=3,
    linewidths=0,
)

ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$\pi_s$")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.set_xlim(left=sizes[0])
ax.set_yscale("log")

# ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")

plt.tight_layout()
plt.savefig("output/small_components.pdf", bbox_inches="tight")
plt.show()
