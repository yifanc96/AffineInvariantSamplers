import numpy as np
import matplotlib.pyplot as plt

def probability_density(x1, x2, epsilon):
    """
    Calculate the probability density function
    π(x) ∝ exp(-(x₁-x₂)²/2ϵ - (x₁+x₂)²/2)
    """
    term1 = -((x1 - x2)**2) / (2 * epsilon)
    term2 = -((x1 + x2)**2) / 2
    return np.exp(term1 + term2)

# Create a grid of x1, x2 values
x_min, x_max = -2, 2
n_points = 1000
x1 = np.linspace(x_min, x_max, n_points)
x2 = np.linspace(x_min, x_max, n_points)
X1, X2 = np.meshgrid(x1, x2)

# Set epsilon value
epsilon = 0.01  # You can change this value to see different levels of anisotropy

# Calculate the probability density
Z = probability_density(X1, X2, epsilon)

# Create the contour plot
plt.figure(figsize=(8, 7))
contour = plt.contour(X1, X2, Z, levels=10, colors='black', linewidths=0.5)
contourf = plt.contourf(X1, X2, Z, levels=20)
# plt.colorbar(contourf)

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
# plt.title(f'Contour Plot of $\\pi(x) \\propto \\exp\\left(-\\frac{{(x_1-x_2)^2}}{{2\\epsilon}} - \\frac{{(x_1+x_2)^2}}{{2}}\\right)$ with $\\epsilon = {epsilon}$')
# plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')  # Ensure the aspect ratio is equal
plt.axis('off')
# Add axes
# plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
# plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.tight_layout()
plt.savefig("anisotropic-plot.pdf")
plt.show()
