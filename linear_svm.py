import numpy as np
from scipy.optimize import minimize

class LinearSVM:
    def __init__(self):
        self.w = None
        self.b = None
        self.support_vectors = None

    def fit(self, x, y):
        def objective(z):
            return 0.5 * np.dot(z[:2], z[:2])

        def jac(z):
            return np.array([z[0], z[1], 0.0])

        constraints = []
        for xi, yi in zip(x, y):
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, xi=xi, yi=yi: yi * (np.dot(z[:2], xi) + z[2]) - 1.0,
                'jac': lambda z, xi=xi, yi=yi: np.array([yi * xi[0], yi * xi[1], yi])
            })

        result = minimize(
            objective, x0=np.array([1.0, 1.0, -5.0]), jac=jac,
            constraints=constraints, method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if not result.success:
            raise RuntimeError(result.message)

        self.w = result.x[:2]
        self.b = result.x[2]
        functional_margin = y * (x @ self.w + self.b)
        mask = np.isclose(functional_margin, 1.0, atol=1e-4)
        self.support_vectors = x[mask]
        return self

    def decision_function(self, x):
        return x @ self.w + self.b

    def predict(self, x):
        scores = self.decision_function(x)
        return np.where(scores >= 0, 1, -1)

    def get_margin(self):
        return 2.0 / np.linalg.norm(self.w)

    def separator_equation(self):
        return f"{self.w[0]:.4f} * x1 + {self.w[1]:.4f} * x2 + {self.b:.4f} = 0"
