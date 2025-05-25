import numpy as np
import matplotlib.pyplot as plt


X0 = np.array([0.3, 0.3])
T = 10  # Time horizon [s]
dt = 0.1  # Time step [s]
N = int(T / dt)  # Number of time steps

# Cost weights
Q = np.diag([0.01, 0.01])
R = np.diag([0.001, 0.001])
Qf = np.diag([5.0, 5.0])


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


def dynamics(x, u, order: int = 1):
    # first-order dynamic system:
    # x_dot(t) = f(x(t), u(t)) = u(t)
    # second-order dynamics:
    # x_dot(t) = f(x(t), u(t)) = [u(t)[0] * x(t)[1], u(t)[1] * x(t)[0]]
    if order == 1:
        return u
    elif order == 2:
        return np.array([u[0] * x[1], u[1] * x[0]])
    else:
        raise ValueError("Order must be 1 or 2.")


def rollout(x0, U):
    X = np.zeros((N+1, len(x0)))
    X[0] = x0
    for k in range(N):
        X[k+1] = X[k] + dt * dynamics(X[k], U[k])
    return X


def runge_kutta(x0, U):
    X = np.zeros((N+1, len(x0)))
    X[0] = x0
    for k in range(N):
        k1 = dynamics(X[k], U[k])
        k2 = dynamics(X[k] + dt / 2 * k1, U[k])
        k3 = dynamics(X[k] + dt / 2 * k2, U[k])
        k4 = dynamics(X[k] + dt * k3, U[k])
        X[k+1] = X[k] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
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
    # 1) compute φ_k on a quadrature grid
    K = 5           # max frequency in each direction
    M = X.shape[0]
    # X[:,0] → shape (M,1,1), broadcast against i,j
    xv = X[:, 0].reshape(M, 1, 1)
    yv = X[:, 1].reshape(M, 1, 1)
    ix = np.arange(K+1).reshape(1, K+1, 1)
    iy = np.arange(K+1).reshape(1, 1, K+1)
    return np.cos(np.pi*ix*xv) * np.cos(np.pi*iy*yv)


def dphi_dx(pt):
    K = 5  # max frequency in each direction
    ix = np.arange(K+1)
    iy = np.arange(K+1)
    dpx = -np.pi*ix * \
        np.sin(np.pi*ix*pt[0])[:, None] * np.cos(np.pi*iy*pt[1])[None, :]
    dpy = -np.pi*iy * \
        np.cos(np.pi*ix*pt[0])[:, None] * np.sin(np.pi*iy*pt[1])[None, :]
    return np.stack([dpx, dpy], axis=0)  # shape (2,K+1,K+1)


def compute_target_coeffs(rho, grid_size=100):
    # 2) compute c_k^* on a quadrature grid
    # builimport numpy as npd grid points
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


def compute_total_cost(X, U, R, dt, T):
    # control cost
    cost_u = np.sum(U @ R * U) * dt

    # ergodic coefficients on trajectory
    Phi_traj = phi(X[:-1])     # shape (N,K+1,K+1)
    c_traj = (dt/T)*Phi_traj.sum(axis=0)
    beta = 1000.0
    erg = np.sum(Lambda*(c_traj - c_star)**2)
    return cost_u + (beta*erg), c_traj


def discrete_ilqr(x0, U_init, max_iter=10):
    U = U_init.copy()
    alpha = 1.0
    cost_history = []
    for it in range(max_iter):
        # forward rollout + cost
        X = rollout(x0, U)
        cost0, _ = compute_total_cost(X, U, R, dt, T)
        cost_history.append(cost0)

        # preallocate gains
        k_ff = np.zeros_like(U)
        K_fb = np.zeros((N, 2, 2))

        # compute c_traj once for derivative
        _, c_traj = compute_total_cost(X, U, R, dt, T)

        # backward pass
        Vx = np.zeros(2)
        Vxx = np.zeros((2, 2))
        for k in reversed(range(N)):
            A, B = linearize(X[k], U[k])
            Delta = c_traj - c_star          # shape (K+1,K+1)
            dphi_k = dphi_dx(X[k])          # shape (2, K+1, K+1)

            # Compute ergodic gradient
            # grad_erg = 2 * (dt/T) * np.sum(
            #     Lambda[None, :, :] * Delta[None, :, :] * dphi_k,
            #     axis=(1, 2)
            # )
            grad_erg = 2*(dt/T)*np.array([
                np.sum(Lambda * Delta * dphi_k[0]),    # ∂ℓ/∂x
                np.sum(Lambda * Delta * dphi_k[1])     # ∂ℓ/∂y
            ])

            qx = grad_erg

            qu = 2*R @ U[k]

            Qx = qx + A.T @ Vx
            Qu = qu + B.T @ Vx
            Qxx = A.T @ Vxx @ A   # note: no Q term
            Quu = R + B.T @ Vxx @ B
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

        # line search (now correctly evaluating U_new)
        alpha = 1.0
        for _ in range(10):
            X_nom = rollout(x0, U)
            U_new = U + alpha*k_ff \
                + np.einsum('kij,kj->ki', K_fb,
                            X_nom[:-1])
            X_new = rollout(x0, U_new)
            cost_new, _ = compute_total_cost(X_new, U_new, R, dt, T)
            if cost_new < cost0:
                U = U_new
                break
            alpha *= 0.5

    return rollout(x0, U), U, cost_history


def ergodic_control(initial_condition, T, deltaTime):
    # Initialize the control input
    U = np.zeros((N, 2))
    # Initialize the state
    x = initial_condition.copy()
    # Loop over the time horizon
    for t in np.arange(0, T, deltaTime):
        # Compute the control input based on the current state
        U[int(t/deltaTime)] = np.array([0.5, 0.5])  # Placeholder control input
        # Update the state based on the dynamics and control input
        x += dynamics(x, U[int(t/deltaTime)]) * deltaTime
    return U


c_traj = np.zeros((N+1, 6, 6))  # shape (N+1, K+1, K+1)
K = 5
Lambda = 1.0 / (1 + np.add.outer(np.arange(K+1)**2, np.arange(K+1)**2))
c_star = compute_target_coeffs(combined_distribution)
# U0 = ergodic_control(X0, T, dt)
U0 = np.zeros((N, 2))
X_opt, U_opt, cost_hist = discrete_ilqr(X0, U0, max_iter=10)
trajectory = X_opt[:, :2]  # shape (N+1, 2)

# Combined figure with two subplots: (1) Gaussian distributions & trajectory, (2) Cost history
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

# Plot 1: Gaussian distributions and trajectory
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = combined_distribution(
    np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
contour = ax1.contourf(X, Y, Z, levels=6, alpha=1.0, cmap='viridis')
ax1.scatter(X0[0], X0[1], color='blue', label='Initial Condition')
ax1.plot(trajectory[:, 0], trajectory[:, 1], color='red',
         linewidth=2, label='Ergodic Trajectory')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Gaussian Mixture & Ergodic Trajectory')
ax1.legend(loc='upper right')

# Plot 2: Control inputs [U] over time
ax2.plot(np.arange(N), U_opt[:, 0], label='U1 (x control)', color='blue')
ax2.plot(np.arange(N), U_opt[:, 1], label='U2 (y control)', color='green')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Control Input')
ax2.set_title('Control Inputs Over Time')
ax2.legend(loc='upper right')

# Plot 3: Cost history
ax3.plot(np.arange(len(cost_hist)), cost_hist,
         marker='o', color='orange', label='Cost History')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Cost')
ax3.set_title('Cost History')
ax3.legend(loc='upper right')

plt.tight_layout()
plt.suptitle('First-Order Ergodic Control')
plt.subplots_adjust(top=0.85)  # Adjust title position
plt.savefig('first_order_ergodic_control.png')
plt.show()
