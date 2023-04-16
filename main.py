import math
import numpy as np
from lib.util import plot as pt, transform as tf
from lib import ff, omega, state

# def debug_all():
#     if DEBUG:
#         debug("n_steps", n_steps)
#         debug("q_current", q_current)
#         debug("q_tgt", q_tgt)
#         debug("q_err", q_err)
#         debug("euler_angles_current", euler_angles_current)
#         debug("lambdai", lambdai)
#         debug("theta_err", theta_err)
#         debug("disturbance_torque", disturbance_torque)
#         debug("estimated_disturbance_torque", estimated_disturbance_torque)
#         debug("u", u)
#         debug("tmp_l", tmp_l)
#         debug("tmp_r", tmp_r)
#         debug("omega_dot", omega_dot)
#         debug("q_dot", q_dot)
#         debug("omega_current", omega_current)


def debug(name, value):
    print("{}: {}".format(name, value))

DEBUG = 0

I = np.array([
    [1000, 0, 0],
    [0, 60, 0],
    [-7, 0, 1000]
])

I_inv = np.linalg.inv(I)

euler_angles_tgt = np.array([math.radians(-5), math.radians(5), math.radians(10)])  # 目標ｵｲﾗｰ角
euler_angles_init = np.array([math.radians(30), math.radians(10), math.radians(-20)])  # 初期ｵｲﾗｰ角
omega_init = np.array([0.0, 0.0, 0.0])  # 初期角速度
omega_tgt = np.array([0.0, 0.0, 0.0])  # 初期角速度

# ｸｫｰﾀﾆｵﾝに変換
q_current = tf.euler_to_quaternion(*euler_angles_init)
q_tgt = tf.euler_to_quaternion(*euler_angles_tgt)
q_err = tf.quaternion_multiply(tf.quaternion_inverse(q_current), q_tgt)
omega_current = omega_init

# ｼﾐｭﾚｰｼｮﾝﾊﾟﾗﾒｰﾀ
n_steps = 5000
dt = 0.1

# plot用変数の生成
q_current_his, q_err_his, theta_err_his, euler_angles_current_his, lambda_his = [], [], [], [], []
estimated_disturbance_torque_his = []

# class生成
ff = ff.DisturbTorq()
omega = omega.AngulVelocity(omega_current, omega_tgt)
state = state.Quaternion(q_current, q_tgt)


for step in range(n_steps):
    # 外乱ﾄﾙｸの算出
    # u = ff.calc_disturb_trq(q_err, omega_current, step)
    u = ff.calc_disturb_trq(q_err, omega.omega_err, step)
    # 角速度の算出
    omega_current = omega.calc_omega(I, omega_current, u, I_inv, dt)
    # ｸｫｰﾀﾆｵﾝの算出
    q_current, q_err = state.calc_quater(omega_current, dt)

    # debug用に座標変換を実施
    euler_angles_current = tf.quaternion_to_euler(q_current) # ｸｫｰﾀﾆｵﾝからｵｲﾗｰ角を計算
    lambdai, theta_err = tf.quaternion_to_axis_angle(q_err) # ｸｫｰﾀﾆｵﾝ誤差から回転単位ﾍﾞｸﾄﾙ、角度を計算

    # 配列に各要素を追加
    q_current_his.append(q_current.copy())
    q_err_his.append(q_err)
    euler_angles_current_his.append(euler_angles_current.copy())
    lambda_his.append(lambdai)
    estimated_disturbance_torque_his.append(u)

# ﾌﾟﾛｯﾄを実施
pt.plot(q_current_his, q_err_his, euler_angles_current_his, lambda_his, n_steps)
pt.plot2(estimated_disturbance_torque_his,  n_steps)