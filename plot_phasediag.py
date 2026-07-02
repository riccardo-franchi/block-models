from numpy import loadtxt, arange
import matplotlib.pyplot as plt
import matplotlib as mpl

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

# Parameters
x = 0.25  # Mixing parameter
IMPOSSIBLE = 0.05
VRANGE = 0.4

# Read the data
img = loadtxt("output/phasediag.txt", float)
Lx, Ly = img.shape

# Adjust the negative values
for i in range(Lx):
    for j in range(Ly):
        if img[i, j] < 0.0:
            img[i, j] = -IMPOSSIBLE

# Make the plot
plt.imshow(
    img,
    vmin=-VRANGE,
    vmax=VRANGE,
    origin="lower",
    interpolation="bilinear",
    cmap="seismic",
    extent=[0, 1, 0, 1],
)

plt.axhline(0.25, linestyle="--", color="k", zorder=1)
plt.axvline(0.25, linestyle="--", color="k", zorder=1)
xpoints = arange(0, 0.7, 0.01)
ypoints = list()
for xval in xpoints:
    ypoints.append(0.75 - 1 / (12 - 16 * xval))
plt.plot(xpoints, ypoints, "k--")

# Label the plot
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r"Group 1 mean degree $c_1$", labelpad=10)
plt.ylabel(r"Group 2 mean degree $c_2$", labelpad=10)
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.tick_params(direction="in", top=True, right=True)
plt.text(0.65, 0.7, "Giant component exists", ha="center", va="center", fontsize=11)
plt.text(0.4, 0.4, "No giant\ncomponent", ha="center", va="center", fontsize=11)
plt.text(0.12, 0.45, "Impossible", ha="center", va="center", rotation=90, fontsize=11)
plt.text(0.45, 0.12, "Impossible", ha="center", va="center", fontsize=11)

plt.savefig("output/phasediag2.pdf", bbox_inches="tight")

plt.show()
