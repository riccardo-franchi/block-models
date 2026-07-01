import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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


def boundary(x):
    return 0.25 * (3 + 1 / (4 * x - 3))


x_zero = 2 / 3  # boundary(x_zero) == 0

x_curve = np.linspace(0, x_zero, 500)
y_curve = boundary(x_curve)

x_fill = np.linspace(0.25, 1, 500)
lower_fill = np.where(x_fill <= x_zero, np.maximum(boundary(x_fill), 0.25), 0.25)

color = "#2166ac"
grey = "0.85"

fig, ax = plt.subplots(figsize=(5.0, 5.0))

ax.axvspan(
    0,
    0.25,
    ymin=0.25,
    facecolor="none",
    edgecolor=grey,
    hatch="///",
    zorder=-1,
    linewidth=0,
)
ax.axhspan(
    0, 0.25, facecolor="none", edgecolor=grey, hatch="///", zorder=-1, linewidth=0
)


ax.fill_between(x_fill, lower_fill, 1, color=color, alpha=0.25, zorder=0, linewidth=0)

ax.plot(x_curve, y_curve, color=color, linestyle="--", zorder=2)
ax.axhline(0.25, color=color, linestyle="--", zorder=1)
ax.axvline(0.25, color=color, linestyle="--", zorder=1)

ax.text(0.65, 0.7, "Giant component exists", ha="center", va="center", fontsize=11)
ax.text(0.4, 0.4, "No giant\ncomponent", ha="center", va="center", fontsize=11)
ax.text(0.12, 0.45, "Impossible", ha="center", va="center", rotation=90, fontsize=11)
ax.text(0.45, 0.12, "Impossible", ha="center", va="center", fontsize=11)

ax.set_xlabel(r"Group 1 mean degree $c_1$")
ax.set_ylabel(r"Group 2 mean degree $c_2$")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("output/phasediag.pdf", bbox_inches="tight")
plt.show()
