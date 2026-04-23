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

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(sizes, analytical, color="#2166ac", label="Analytical", zorder=1)
ax.scatter(
    sizes,
    simulated,
    s=6,
    color="#d6604d",
    label="Simulation",
    zorder=2,
    linewidths=0,
)

ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$\pi_s$")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.set_xlim(left=sizes[0])
ax.set_yscale("log")

ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")

plt.tight_layout()
plt.savefig("output/small_components.svg", bbox_inches="tight")
plt.show()
