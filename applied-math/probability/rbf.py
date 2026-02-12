import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(x1, x2, lengthscale=1.0):
    """
    Compute RBF (Gaussian) kernel between inputs.
    
    K(x1, x2) = exp(-||x1 - x2||^2 / (2 * lengthscale^2))
    """
    # Squared Euclidean distance
    # Shape: (n1, n2) where n1 = len(x1), n2 = len(x2)
    sq_dist = np.sum(x1**2, axis=1, keepdims=True) + \
              np.sum(x2**2, axis=1) - \
              2 * x1 @ x2.T
    
    return np.exp(-sq_dist / (2 * lengthscale**2))


# Example: visualize kernel as function of distance
d = np.linspace(0, 4, 100).reshape(-1, 1)
zeros = np.zeros_like(d)

plt.figure(figsize=(10, 4))

# Left plot: kernel vs distance for different lengthscales
plt.subplot(1, 2, 1)
for ls in [0.5, 1.0, 2.0]:
    k = rbf_kernel(d, zeros, lengthscale=ls)
    plt.plot(d, k, label=f"ℓ = {ls}")
plt.xlabel("Distance ||x - x'||")
plt.ylabel("K(x, x')")
plt.title("RBF Kernel Decay")
plt.legend()
plt.grid(alpha=0.3)

# Right plot: kernel matrix heatmap
plt.subplot(1, 2, 2)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
K = rbf_kernel(X, X, lengthscale=1.0)
plt.imshow(K, cmap="viridis", origin="lower", extent=[-3, 3, -3, 3])
plt.colorbar(label="K(x, x')")
plt.xlabel("x'")
plt.ylabel("x")
plt.title("RBF Kernel Matrix")

plt.tight_layout()
plt.show()

class RBFNetwork:
    """
    Simple RBF Network for function approximation.
    
    Model: f(x) = Σ w_i * φ(||x - c_i||)
    """
    
    def __init__(self, n_centers, lengthscale=1.0):
        self.n_centers = n_centers
        self.lengthscale = lengthscale
        self.centers = None
        self.weights = None
    
    def fit(self, X, y):
        """Fit RBF network to data."""
        # Choose centers (can use k-means, but here: random subset)
        idx = np.random.choice(len(X), self.n_centers, replace=False)
        self.centers = X[idx]
        
        # Compute design matrix: Φ[i,j] = rbf(X[i], centers[j])
        Phi = rbf_kernel(X, self.centers, self.lengthscale)
        
        # Solve for weights: (Φ^T Φ)^{-1} Φ^T y
        self.weights = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return self
    
    def predict(self, X):
        """Predict at new points."""
        Phi = rbf_kernel(X, self.centers, self.lengthscale)
        return Phi @ self.weights


# Demo: approximate a nonlinear function
np.random.seed(42)
X_train = np.linspace(-3, 3, 20).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(20)

X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

model = RBFNetwork(n_centers=10, lengthscale=0.5).fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, c="red", s=50, label="Training data", zorder=3)
plt.plot(X_test, y_true, "k--", label="True: sin(x)", alpha=0.5)
plt.plot(X_test, y_pred, "b-", linewidth=2, label="RBF prediction")
plt.scatter(model.centers, np.zeros(len(model.centers)), 
            c="green", s=100, marker="x", label="RBF centers", zorder=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("RBF Network: Function Approximation")
plt.grid(alpha=0.3)
plt.show()