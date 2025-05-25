import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([0.3, 0.3])
T = 10  # Time horizon [s]
dt = 0.1  # Time step [s]
N = int(T / dt)  # Number of time steps

# Cost weights
Q = np.diag([0.01, 0.01])
R = np.diag([0.001, 0.001])
Qf = np.diag([1.0, 2.0])


class GaussianDistribution:
    def __init__(self, omega, mu, sigma):
        self.omega = omega
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x):
        # x: shape (N, 2)
        d = x - self.mu  # shape (N, 2)
        inv_sigma = np.linalg.inv(self.sigma)
        exponent = np.einsum('ni,ij,nj->n', d, inv_sigma, d)
        norm_const = np.sqrt(np.linalg.det(2 * np.pi * self.sigma))
        return self.omega * np.exp(-0.5 * exponent) / norm_const


G1 = GaussianDistribution(
    omega=0.5, mu=np.array([0.35, 0.38]).T,
    sigma=np.array([[0.01, 0.004], [0.004, 0.01]])
)
G2 = GaussianDistribution(
    omega=0.2, mu=np.array([0.68, 0.25]).T,
    sigma=np.array([[0.005, -0.003], [-0.003, 0.005]])
)
G3 = GaussianDistribution(
    omega=0.3, mu=np.array([0.56, 0.64]).T,
    sigma=np.array([[0.008, 0], [0, 0.004]])
)

# Dynamics and linearization


def dynamics(x, u):
    # 2D first-order dynamic system:
    # x_dot(t) = f(x(t), u(t)) = u(t)
    return u


def rollout(x0, U):
    X = np.zeros((N+1, len(x0)))
    X[0] = x0
    for k in range(N):
        X[k+1] = X[k] + dt * dynamics(X[k], U[k])
    return X


def linearize(x, u):
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[dt, 0], [0, dt]])
    return A, B


def phi(X):
    """
    Given X of shape (M,2), returns Phi of shape (M, K+1, K+1)
    where Phi[m,i,j] = cos(pi*i * X[m,0]) * cos(pi*j * X[m,1])
    """
    K = 5           # max frequency in each direction
    M = X.shape[0]
    # X[:,0] → shape (M,1,1), broadcast against i,j
    xv = X[:, 0].reshape(M, 1, 1)
    yv = X[:, 1].reshape(M, 1, 1)
    ix = np.arange(K+1).reshape(1, K+1, 1)
    iy = np.arange(K+1).reshape(1, 1, K+1)
    return np.cos(np.pi*ix*xv) * np.cos(np.pi*iy*yv)


def dphi_dx(pt):
    """Compute gradient of phi with respect to x at point pt"""
    K = 5  # max frequency in each direction
    ix = np.arange(K+1)
    iy = np.arange(K+1)
    dpx = -np.pi*ix * \
        np.sin(np.pi*ix*pt[0])[:, None] * np.cos(np.pi*iy*pt[1])[None, :]
    dpy = -np.pi*iy * \
        np.cos(np.pi*ix*pt[0])[:, None] * np.sin(np.pi*iy*pt[1])[None, :]
    return np.stack([dpx, dpy], axis=0)  # shape (2,K+1,K+1)


def compute_target_coeffs(rho, grid_size=100):
    """Compute target ergodic coefficients"""
    lin = np.linspace(0, 1, grid_size)
    Xg, Yg = np.meshgrid(lin, lin)
    XY = np.vstack([Xg.ravel(), Yg.ravel()]).T          # (M,2), M=grid_size^2
    R = rho(XY)                                       # (M,)
    Phi = phi(XY)                                      # (M,K+1,K+1)
    # approximate integral ∫ φ ρ dx ≈ (1/M) Σ φ*ρ
    c_star = (Phi * R[:, None, None]).sum(axis=0) / R.size
    return c_star


def combined_distribution(x):
    return G1.pdf(x) + G2.pdf(x) + G3.pdf(x)


def compute_total_cost(X, U, c_star, Lambda, R, dt, T):
    """
    Compute total cost including ergodic and control terms
    """
    # control cost
    cost_u = np.sum([U[k].T @ R @ U[k] for k in range(len(U))]) * dt

    # ergodic coefficients on trajectory
    Phi_traj = phi(X[:-1])     # shape (N,K+1,K+1)
    c_traj = (dt/T) * Phi_traj.sum(axis=0)

    # ergodic cost
    erg = np.sum(Lambda * (c_traj - c_star)**2)
    return cost_u + erg, c_traj


def discrete_ilqr(x0, U_init, c_star, Lambda, max_iter=10):
    U = U_init.copy()
    alpha = 1.0
    cost_history = []

    for it in range(max_iter):
        # forward rollout + cost
        X = rollout(x0, U)
        cost0, c_traj = compute_total_cost(X, U, c_star, Lambda, R, dt, T)
        cost_history.append(cost0)

        print(f"Iteration {it}: Cost = {cost0:.6f}")

        # preallocate gains
        k_ff = np.zeros_like(U)
        K_fb = np.zeros((N, 2, 2))

        # backward pass
        Vx = np.zeros(2)
        Vxx = np.zeros((2, 2))

        for k in reversed(range(N)):
            A, B = linearize(X[k], U[k])

            # ergodic gradient at stage k
            Delta = c_traj - c_star          # shape (K+1,K+1)
            dphi_k = dphi_dx(X[k])          # shape (2, K+1, K+1)

            # Compute ergodic gradient properly
            grad_erg = 2 * (dt/T) * np.sum(
                Lambda[None, :, :] * Delta[None, :, :] * dphi_k,
                axis=(1, 2)
            )

            qx = grad_erg
            qu = 2 * R @ U[k]

            Qx = qx + A.T @ Vx
            Qu = qu + B.T @ Vx
            Qxx = A.T @ Vxx @ A
            Quu = 2 * R + B.T @ Vxx @ B  # Fixed: include R term
            Qux = B.T @ Vxx @ A

            # Regularization for numerical stability
            Quu += 1e-6 * np.eye(2)

            try:
                invQuu = np.linalg.inv(Quu)
            except np.linalg.LinAlgError:
                invQuu = np.linalg.pinv(Quu)

            k_ff[k] = -invQuu @ Qu
            K_fb[k] = -invQuu @ Qux

            Vx = Qx + K_fb[k].T @ Quu @ k_ff[k] + \
                K_fb[k].T @ Qu + Qux.T @ k_ff[k]
            Vxx = Qxx + K_fb[k].T @ Quu @ K_fb[k] + \
                Qux.T @ K_fb[k] + K_fb[k].T @ Qux

        # line search
        alpha = 1.0
        for ls_iter in range(10):
            X_nom = rollout(x0, U)
            dX = X_nom[:-1] - X[:-1]  # deviation from nominal
            U_new = U + alpha * k_ff + \
                np.array([K_fb[k] @ dX[k] for k in range(N)])
            X_new = rollout(x0, U_new)
            cost_new, _ = compute_total_cost(
                X_new, U_new, c_star, Lambda, R, dt, T)

            if cost_new < cost0:
                U = U_new
                break
            alpha *= 0.5
        else:
            print(f"Line search failed at iteration {it}")

        # Check convergence
        if it > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-6:
            print(f"Converged at iteration {it}")
            break

    return rollout(x0, U), U, cost_history


def ergodic_control(initial_condition, T, deltaTime):
    """Initialize control with simple policy"""
    U = np.random.randn(N, 2) * 0.1  # Small random initialization
    return U


# Setup ergodic coefficients and weights
K = 5
c_star = compute_target_coeffs(combined_distribution)
Lambda = 1.0 / (1 + np.add.outer(np.arange(K+1)**2, np.arange(K+1)**2))

# Perform Ergodic control from the initial condition
U0 = ergodic_control(X0, T, dt)

# Run iLQR to optimize the control input
print("Running iLQR optimization...")
X_opt, U_opt, cost_hist = discrete_ilqr(X0, U0, c_star, Lambda, max_iter=20)

# Extract the trajectory
trajectory = X_opt[:, :2]  # shape (N+1, 2)

# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Gaussian distributions and trajectory
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = combined_distribution(
    np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

ax1.contourf(X, Y, Z, levels=10, alpha=0.8, cmap='viridis')
ax1.scatter(X0[0], X0[1], color='blue', s=100,
            label='Initial Condition', zorder=5)
ax1.plot(trajectory[:, 0], trajectory[:, 1], color='red', linewidth=2,
         label='Ergodic Trajectory', zorder=4)
ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], color='orange', s=100,
            label='Final Position', zorder=5)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Ergodic Control over Gaussian Mixture')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cost history
ax2.plot(cost_hist, 'b-o', linewidth=2, markersize=4)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Total Cost')
ax2.set_title('Cost Convergence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final statistics
final_cost, final_c_traj = compute_total_cost(
    X_opt, U_opt, c_star, Lambda, R, dt, T)
print(f"\nFinal cost: {final_cost:.6f}")
print(f"Trajectory length: {len(trajectory)} points")
print(f"Final position: [{trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f}]")
