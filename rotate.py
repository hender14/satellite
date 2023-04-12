import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm == 0 or np.isnan(norm):
        raise ValueError("Quaternion has zero or NaN norm, cannot normalize")
    return q / norm

def euler_to_rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    C = np.array([[ctheta * cpsi, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
                  [ctheta * spsi, sphi * stheta * spsi + cphi * cpsi, cphi * stheta * spsi - sphi * cpsi],
                  [-stheta, sphi * ctheta, cphi * ctheta]]).T
    return C

def rotation_matrix_to_quaternion(C):
    q4 = 0.5 * np.sqrt(1 + C[0, 0] + C[1, 1] + C[2, 2])
    # q1 = 0.25 * (C[2, 1] - C[1, 2]) / q4 #後で戻す
    # q2 = 0.25 * (C[0, 2] - C[2, 0]) / q4
    # q3 = 0.25 * (C[1, 0] - C[0, 1]) / q4
    q1 = 0.25 * (C[1, 2] - C[2, 1]) * q4
    q2 = 0.25 * (C[2, 0] - C[0, 2]) * q4
    q3 = 0.25 * (C[0, 1] - C[1, 0]) * q4
    return np.array([q1, q2, q3, q4])

def euler_to_quaternion(phi, theta, psi):
    C = euler_to_rotation_matrix(phi, theta, psi)
    # print("C = ", C)
    q = rotation_matrix_to_quaternion(C)
    # print("q = ", q)
    return q

def quaternion_to_euler(q):
    C = quaternion_to_rotation_matrix(q)
    euler_angle = rotation_matrix_to_euler(C)
    return euler_angle

def quaternion_to_axis_angle(q):
    C = quaternion_to_rotation_matrix(q)
    # print(f"C = {C}")
    lambdai, theta_err = rotation_matrix_to_axis_angle(C)
    return lambdai, theta_err

def quaternion_to_rotation_matrix(q):
    # Convert quaternion to rotation matrix
    q1, q2, q3, q4 = q
    C = np.array([
        [q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4)],
        [2 * (q1 * q2 - q3 * q4), -q1 ** 2 + q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q2 * q3 + q1 * q4)],
        [2 * (q1 * q3 + q2 * q4), 2 * (q2 * q3 - q1 * q4), -q1 ** 2 - q2 ** 2 + q3 ** 2 + q4 ** 2]
    ])
    return C

def rotation_matrix_to_euler(C):
    phi = np.arctan2(C[1, 2], C[2, 2])
    theta = np.arctan2(-C[0, 2], np.sqrt(C[1, 2]**2 + C[2, 2]**2))
    psi = np.arctan2(C[0, 1], C[0, 0])

    return np.array([math.degrees(phi), math.degrees(theta), math.degrees(psi)])

def rotation_matrix_to_axis_angle(C):
    # Compute the rotation angle
    theta = np.arccos(0.5 * ((C[0, 0] + C[1, 1] + C[2, 2]) - 1))
    
    # Compute the axis direction
    sin_theta = np.sin(theta)
    lambda_1 = (C[1, 2] - C[2, 1]) / (2 * sin_theta)
    lambda_2 = (C[2, 0] - C[0, 2]) / (2 * sin_theta)
    lambda_3 = (C[0, 1] - C[1, 0]) / (2 * sin_theta)
    lambdai = np.array([lambda_1, lambda_2, lambda_3])

    return lambdai, theta

def quaternion_multiply(q, p):
    # Multiply two quaternions
    q1, q2, q3, q4 = q
    p1, p2, p3, p4 = p
    # print("p:{}, q:{}".format(p, q))

    r1 = q4 * p1 + q1 * p4 + q3 * p2 - q2 * p3
    r2 = q4 * p2 + q1 * p3 + q2 * p4 - q3 * p1
    r3 = q4 * p3 - q1 * p2 + q2 * p1 + q3 * p4
    r4 = q4 * p4 - q1 * p1 - q2 * p2 - q3 * p3
    # print(f"pq = {np.array([r1, r2, r3, r4])}")

    return np.array([r1, r2, r3, r4])

def quaternion_inverse(q):
    # Calculate the inverse of a quaternion
    q_star = q.copy()
    q_star[:3] = -q_star[:3]
    return q_star #後で戻す
    # return q_star / np.linalg.norm(q)


def tmp_multi(w, tmp):
    w1, w2, w3 = w
    tmp1, tmp2, tmp3 = tmp

    tmp1 = w2*tmp3 - w3*tmp2
    tmp2 = w3*tmp1 - w1*tmp3
    tmp3 = w1*tmp2 - w2*tmp1

    return np.array([tmp1, tmp2, tmp3])

# 入力パラメータと初期条件
alpha = 10.
beta = 50.
I = np.array([
    [1000, 0, 0],
    [0, 60, 0],
    [-7, 0, 1000]
])

I_inv = np.array([
    [0.001, 0, 0],
    [0, 0.0166666666666667, 0],
    [0.0000065, 0, 0.001]
])

pi = math.pi
euler_angles_tgt = np.array([math.radians(-5), math.radians(5), math.radians(10)])  # 目標オイラー角
euler_angles_init = np.array([math.radians(30), math.radians(10), math.radians(-20)])  # 初期オイラー角
omega_init = np.array([0.0, 0.0, 0.0])  # 初期角速度

# print("euler_angles_init: {}".format(euler_angles_init))
# クォータニオンに変換
q_current = euler_to_quaternion(*euler_angles_init)
# print("q_current: {}".format(q_current))
q_tgt = euler_to_quaternion(*euler_angles_tgt)
omega_current = omega_init

# シミュレーション
n_steps = 5000
dt = 0.1

q_current_his = []
q_err_his = []
theta_err_his = []
euler_angles_current_his = []
lambda_his = []

for step in range(n_steps):
    print("step: {}".format(step))
    print("q_tgt: {}".format(q_tgt))
    q_err = quaternion_multiply(quaternion_inverse(q_current), q_tgt)
    print(f"q_err: {q_err}")

    # クォータニオンからオイラー角を計算
    euler_angles_current = quaternion_to_euler(q_current)
    print(f"euler_angles_current: {euler_angles_current}")

    # クォータニオン誤差から回転単位ベクトル、角度を計算
    lambdai, theta_err = quaternion_to_axis_angle(q_err)
    print("lambdai: {}, theta_err: {}".format(lambdai, theta_err))

    u = alpha * q_err[:3] - beta * omega_current
    print("omega_current: {}".format(omega_current))
    print(f"u = {u}")

    # Solve the motion equations
    tmp_l = (I @ omega_current.T).T
    print("tmp_l: {}".format(tmp_l))
    # print("tmp_: {}".format(omega_current @ tmp_l))
    tmp_r = u - tmp_multi(omega_current, tmp_l)
    print("tmp_r: {}".format(tmp_r))
    omega_dot = (I_inv @ tmp_r.T).T
    q_dot = 0.5 * quaternion_multiply(q_current, np.append(omega_current, 0))
    print("q_dot: {}".format(q_dot))
    print("omega_dot: {}".format(omega_dot))
    # クォータニオンと角速度を更新
    q_current += q_dot * dt
    omega_current += omega_dot * dt
    print("omega_current: {}".format(omega_current))

    q_current_his.append(q_current.copy())
    q_err_his.append(q_err)
    euler_angles_current_his.append(euler_angles_current.copy())
    lambda_his.append(lambdai)
    # theta_err_his.append(theta_err)

time = np.linspace(0, n_steps/10, n_steps)
quaternions = np.array(q_current_his)
error_quaternions = np.array(q_err_his)
# theta_error_quaternions = np.array(theta_err_his)
euler_angles = np.array(euler_angles_current_his)
rotation_axes = np.array(lambda_his)

# グラフの初期化
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 時間とクォータニオンの関係
axs[0, 0].plot(time, quaternions)
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Quaternions')

# 時間と誤差クォータニオンの関係
axs[0, 1].plot(time, error_quaternions)
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Error Quaternions')

# 時間とオイラー角の関係
axs[1, 0].plot(time, euler_angles)
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Euler Angles')

# 時間と回転軸の変化の関係
# axs[1, 1].plot(time, theta_error_quaternions)
axs[1, 1].plot(time, rotation_axes)
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Rotation Axes')

# グラフのレイアウトを調整
fig.tight_layout()

# グラフをPNGファイルに保存
fig.savefig('out/combined_graph.png')