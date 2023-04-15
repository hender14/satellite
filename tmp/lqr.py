import numpy as np
from scipy import linalg

# Constants and parameters
I = np.array([[1000, 0, 0],
              [0, 60, 0],
              [0, 0, 1000]])  # Inertia matrix
A = np.zeros((6, 6))  # State matrix (3x3 quaternion error + 3x3 angular velocity error)
A[3:, :3] = -np.linalg.inv(I)  # Fill the lower-left block with the negative inverse of the inertia matrix
B = np.vstack((np.zeros((3, 3)), np.linalg.inv(I)))  # Input matrix (6x3, with the lower-right block being the inverse of the inertia matrix)

# LQR weights
Q = np.diag([1, 1, 1, 1e-2, 1e-2, 1e-2])  # State weight matrix (emphasize quaternion error over angular velocity error)
R = np.diag([1e-3, 1e-3, 1e-3])  # Control input weight matrix (minimize torque)

# Calculate the LQR gain matrix K
P = linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Simulation parameters
n_steps = 100
dt = 0.1

# Initial conditions
q_err_init = np.array([0.1, 0.1, 0.1])  # Initial quaternion error
omega_err_init = np.array([0.01, 0.01, 0.01])  # Initial angular velocity error
state = np.hstack((q_err_init, omega_err_init))  # Initial state

# Simulation loop
for step in range(n_steps):
    u = -K @ state  # Calculate the control input (torque)
    state_dot = A @ state + B @ u  # Update the state derivative
    state += state_dot * dt  # Update the state
