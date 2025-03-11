import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import gamma
import imageio
import os

output_dir = "dirichlet_frames"
os.makedirs(output_dir, exist_ok=True)

def dirichlet_pdf(x, y, alphas):
    """
    Computes the Dirichlet PDF for a 3-dimensional Dirichlet distribution.
    Given x and y, the third coordinate is 1 - x - y.
    Only valid for points in the simplex: x >= 0, y >= 0, and x+y <= 1.
    """
    alpha1, alpha2, alpha3 = alphas
    # Calculate z coordinate in the simplex
    z = 1 - x - y
    # Only calculate where z is non-negative
    valid = z >= 0
    pdf = np.zeros_like(x)

    # Normalization constant
    alpha0 = alpha1 + alpha2 + alpha3
    B = gamma(alpha1) * gamma(alpha2) * gamma(alpha3) / gamma(alpha0)

    # Compute PDF only for valid points
    pdf[valid] = (1 / B) * (x[valid] ** (alpha1 - 1)) * (y[valid] ** (alpha2 - 1)) * (z[valid] ** (alpha3 - 1))
    return pdf

# Create a meshgrid on the simplex domain (x, y such that x+y <= 1)
grid_points = 200
x = np.linspace(0, 1, grid_points)
y = np.linspace(0, 1, grid_points)
X, Y = np.meshgrid(x, y)
mask = (X + Y) <= 1
X_masked = np.where(mask, X, np.nan)
Y_masked = np.where(mask, Y, np.nan)

alpha_values = np.linspace(2, 0.1, 40)  # Symmetric alphas changing simultaneously

filenames = []

for i, alpha in enumerate(alpha_values):
    alphas = (alpha, alpha, alpha)  # Symmetric prior
    Z = dirichlet_pdf(X, Y, alphas)
    Z = np.where(mask, Z, np.nan)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X_masked, Y_masked, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8)
    ax.set_title(
        r"Dirichlet PDF on the 2-simplex: $\alpha_1=\alpha_2=\alpha_3={:.2f}$".format(alpha),
        fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')

    ax.view_init(elev=45, azim=30)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    filename = os.path.join(output_dir, f"frame_{i:03d}.png")
    plt.savefig(filename, dpi=100)
    filenames.append(filename)
    plt.close()

gif_filename = "plots/dirichlet_animation_symmetric.gif"
with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"Animation saved as {gif_filename}")
