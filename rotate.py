import math
import numpy as np
import plot as pt
import transform as tf

def debug_all():
    if DEBUG:
        debug("n_steps", n_steps)
        debug("q_current", q_current)
        debug("q_tgt", q_tgt)
        debug("q_err", q_err)
        debug("euler_angles_current", euler_angles_current)
        debug("lambdai", lambdai)
        debug("theta_err", theta_err)
        debug("disturbance_torque", disturbance_torque)
        debug("estimated_disturbance_torque", estimated_disturbance_torque)
        debug("u", u)
        debug("tmp_l", tmp_l)
        debug("tmp_r", tmp_r)
        debug("omega_dot", omega_dot)
        debug("q_dot", q_dot)
        debug("omega_current", omega_current)


def debug(name, value):
    print("{}: {}".format(name, value))

# 外乱トルク生成用関数
def generate_disturbance_torque(step, period = 100, amplitude = np.array([0.5, 0.5, 0.5])):
    return amplitude * np.sin(2 * np.pi * step / period) + np.random.normal(0, 0.5, 3)

# 簡易カルマンフィルタクラス
class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_estimate):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = initial_estimate

    def update(self, measurement):
        # 予測ステップ
        prediction = self.estimate

        # 更新ステップ
        kalman_gain = self.process_noise / (self.process_noise + self.measurement_noise)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.process_noise = (1 - kalman_gain) * self.process_noise

        return self.estimate

DEBUG = 0

# カルマンフィルタ初期化
disturbance_torque_filter = SimpleKalmanFilter(0.1, 0.1, np.zeros(3))

# period = 100  # 外乱トルクの周期
# amplitude = np.array([0.5, 0.5, 0.5])  # 外乱トルクの振幅

# 入力パラメータと初期条件
alpha, beta = 10., 50.

I = np.array([
    [1000, 0, 0],
    [0, 60, 0],
    [-7, 0, 1000]
])
# I = np.diag([5000, 6000, 7000])

# I_inv = np.array([
#     [0.001, 0, 0],
#     [0, 0.0166666666666667, 0],
#     [0.0000065, 0, 0.001]
# ])
I_inv = np.linalg.inv(I)

euler_angles_tgt = np.array([math.radians(-5), math.radians(5), math.radians(10)])  # 目標オイラー角
euler_angles_init = np.array([math.radians(30), math.radians(10), math.radians(-20)])  # 初期オイラー角
omega_init = np.array([0.0, 0.0, 0.0])  # 初期角速度

# クォータニオンに変換
q_current = tf.euler_to_quaternion(*euler_angles_init)
q_tgt = tf.euler_to_quaternion(*euler_angles_tgt)
omega_current = omega_init

# シミュレーションパラメータ
n_steps = 5000
dt = 0.1

q_current_his, q_err_his, theta_err_his, euler_angles_current_his, lambda_his = [], [], [], [], []
estimated_disturbance_torque_his = []

for step in range(n_steps):
    q_err = tf.quaternion_multiply(tf.quaternion_inverse(q_current), q_tgt)

    # クォータニオンからオイラー角を計算
    euler_angles_current = tf.quaternion_to_euler(q_current)

    # クォータニオン誤差から回転単位ベクトル、角度を計算
    lambdai, theta_err = tf.quaternion_to_axis_angle(q_err)

    # 外乱トルク生成
    disturbance_torque = generate_disturbance_torque(step)

    # 外乱トルク推定
    estimated_disturbance_torque = disturbance_torque_filter.update(disturbance_torque)

    u = alpha * q_err[:3] - beta * omega_current

    # Solve the motion equations
    tmp_l = (I @ omega_current.T).T
    tmp_r = u + estimated_disturbance_torque - tf.tmp_multi(omega_current, tmp_l)
    omega_dot = (I_inv @ tmp_r.T).T
    q_dot = 0.5 * tf.quaternion_multiply(q_current, np.append(omega_current, 0))

    # クォータニオンと角速度を更新
    q_current += q_dot * dt
    omega_current += omega_dot * dt

    # 配列に各要素を追加
    q_current_his.append(q_current.copy())
    q_err_his.append(q_err)
    euler_angles_current_his.append(euler_angles_current.copy())
    lambda_his.append(lambdai)
    estimated_disturbance_torque_his.append(estimated_disturbance_torque)
    # theta_err_his.append(theta_err)
    debug_all()

pt.plot(q_current_his, q_err_his, euler_angles_current_his, lambda_his, n_steps)
pt.plot2(estimated_disturbance_torque_his,  n_steps)