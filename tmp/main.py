import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib as mpl
mpl.use('Agg') # AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt

def inertia_moment_cube(mass, width, height, depth):
    Ix = (1/12) * mass * (height**2 + depth**2)
    Iy = (1/12) * mass * (width**2 + depth**2)
    Iz = (1/12) * mass * (width**2 + height**2)
    inertia_matrix = np.diag([Ix, Iy, Iz])
    return inertia_matrix

def pd_controller(Kp, Kd, angle_error, angle_rate_error):
    # print(angle_error, angle_rate_error)
    torque_output = -Kp * angle_error - Kd * angle_rate_error
    return torque_output

def torque_to_angular_acceleration(torque_output, inertia_matrix):
    angular_acceleration = np.linalg.inv(inertia_matrix) @ torque_output
    return angular_acceleration

def update_euler_angle_rates(current_euler_angle_rates, angular_acceleration, dt):
    new_euler_angle_rates = current_euler_angle_rates + angular_acceleration * dt
    return new_euler_angle_rates

def simulate_sensors(torque_output, current_euler_angle_rates, current_linear_acceleration, dt):
    angular_acceleration = torque_to_angular_acceleration(torque_output, inertia_matrix)
    new_euler_angle_rates = update_euler_angle_rates(current_euler_angle_rates, angular_acceleration, dt)
    new_linear_acceleration = current_linear_acceleration
    return new_euler_angle_rates, new_linear_acceleration

def get_euler_angles(euler_angle_rates, linear_acceleration, current_euler_angles, dt):
    rotation = R.from_euler('XYZ', current_euler_angles)
    rotation_derivative = R.from_rotvec(euler_angle_rates * dt)
    new_rotation = rotation * rotation_derivative
    new_euler_angles = new_rotation.as_euler('XYZ')
    return new_euler_angles

# Satellite inertia matrix
satellite_mass = 50  # kg
satellite_width = 1  # m
satellite_height = 1  # m
satellite_depth = 1  # m
inertia_matrix = inertia_moment_cube(satellite_mass, satellite_width, satellite_height, satellite_depth)

# PD controller gains
Kp = np.array([200, 200, 200])
# Kp = np.array([200, 0.5, 0.5])
# Kd = np.array([0, 0, 0])
Kd = np.array([5, 5, 5])

# Simulation initialization
# 初期オイラー角をランダムに設定（例: -pi/4 から pi/4 の範囲）
euler_angles = np.random.uniform(-np.pi/4, np.pi/4, 3)

# 初期オイラー角速度をランダムに設定（例: -0.1 から 0.1 の範囲）
euler_angle_rates = np.random.uniform(-0.1, 0.1, 3)

# オイラー角偏差とオイラー角速度偏差を計算（目標オイラー角と目標オイラー角速度がゼロの場合）
euler_angle_error = -euler_angles
euler_angle_rate_error = -euler_angle_rates
# euler_angles = np.array([0, 0, 0])  # [roll, pitch, yaw] in radians
# euler_angle_rates = np.array([0, 0, 0])  # [roll_rate, pitch_rate, yaw_rate] in radians/s
linear_acceleration = np.array([0, 0, 0])  # [x, y, z] in m/s^2
# euler_angle_error = np.array([0, 0, 0])
# euler_angle_rate_error = np.array([0, 0, 0])

torque_output_history = []
euler_angles_history = []
euler_angle_rates_history = []

dt = 0.01  # s

for _ in range(1000):  # Simulation iterations
    torque_output = pd_controller(Kp, Kd, euler_angle_error, euler_angle_rate_error)
    euler_angle_rates, linear_acceleration = simulate_sensors(torque_output, euler_angle_rates, linear_acceleration, dt)
    euler_angles = get_euler_angles(euler_angle_rates, linear_acceleration, euler_angles, dt)
    # print(torque_output, euler_angles, euler_angle_rates)

    # Update the Euler angle error and Euler angle rate error
    euler_angle_error = -euler_angles
    euler_angle_rate_error = -euler_angle_rates
    
    torque_output_history.append(torque_output)
    euler_angles_history.append(euler_angles)
    euler_angle_rates_history.append(euler_angle_rates)

    # Update the Euler angle error and Euler angle rate error as needed before moving to the next step

# Convert the recorded history to NumPy arrays
torque_output_history = np.array(torque_output_history)
euler_angles_history = np.array(euler_angles_history)
euler_angle_rates_history = np.array(euler_angle_rates_history)

# Plot the torque output history
plt.figure()
plt.plot(torque_output_history[:, 0], label='Roll torque')
plt.plot(torque_output_history[:, 1], label='Pitch torque')
plt.plot(torque_output_history[:, 2], label='Yaw torque')
plt.xlabel('Time step')
plt.ylabel('Torque output (N*m)')
plt.grid()
plt.legend()
plt.savefig('Torque output_vs_time.png', dpi=300)
plt.show()

# Plot the Euler angles history
plt.figure()
plt.plot(euler_angles_history[:, 0], label='Roll')
plt.plot(euler_angles_history[:, 1], label='Pitch')
plt.plot(euler_angles_history[:, 2], label='Yaw')
plt.xlabel('Time step')
plt.ylabel('Euler angles (rad)')
plt.grid()
plt.legend()
plt.savefig('Euler_angles_vs_time.png', dpi=300)
plt.show()

# Plot the Euler angle rates history
plt.figure()
plt.plot(euler_angle_rates_history[:, 0], label='Roll rate')
plt.plot(euler_angle_rates_history[:, 1], label='Pitch rate')
plt.plot(euler_angle_rates_history[:, 2], label='Yaw rate')
plt.xlabel('Time step')
plt.ylabel('Euler angle rates (rad/s)')
plt.grid()
plt.legend()
plt.savefig('Euler_angle_rates_vs_time.png', dpi=300)

plt.show()