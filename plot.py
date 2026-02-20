import matplotlib.pyplot as plt
import numpy as np


def compute_giant_cluster_size(c, phi=1.0, tolerance=1e-8, max_iterations=1_000):

    S_guess = np.full(c.shape[0], 0.01)  # Initial guess

    for iteration in range(max_iterations):
        S_new = 1 - np.exp(-phi * c @ S_guess)
        if np.linalg.norm(S_new - S_guess, ord=1) < tolerance:
            break
        S_guess = S_new

    return S_guess, iteration


b = 0.01

c = np.array([[2.0, b], [b, 20.0]])

node_distribution = np.array([0.5, 0.5])


phi_values = np.linspace(0, 1, 100)
S_values = []

for phi in phi_values:
    S_vec, _ = compute_giant_cluster_size(c, phi=phi)
    S = np.dot(node_distribution, S_vec)
    S_values.append(S)

S_values = np.array(S_values)


measured_phi, measured_s = [], []
with open("s_phi.txt") as f:
    for line in f:
        p, s_val = map(float, line.split())
        measured_phi.append(p)
        measured_s.append(s_val)

plt.figure(figsize=(10, 6))
plt.plot(phi_values, S_values, linewidth=2, color="blue", label="Analytical")
plt.scatter(measured_phi, measured_s, s=2, color="red", label="Simulation", zorder=2)
plt.grid(True, alpha=0.3)
plt.xlabel(r"$\phi$")
plt.ylabel(r"$S(\phi)$")
plt.title("Giant component size vs edge occupation probability")
plt.tight_layout()
plt.show()
