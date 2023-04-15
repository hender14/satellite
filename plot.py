import numpy as np
import matplotlib as mpl
mpl.use('Agg') # AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt

def plot(q_current_his, q_err_his, euler_angles_current_his, lambda_his, n_steps):

    time = np.linspace(0, n_steps/10, n_steps)
    quaternions = np.array(q_current_his)
    error_quaternions = np.array(q_err_his)
    # theta_error_quaternions = np.array(theta_err_his)
    euler_angles = np.array(euler_angles_current_his)
    rotation_axes = np.array(lambda_his)

    # ｸﾞﾗﾌの初期化
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 時間とｸｫｰﾀﾆｵﾝの関係
    axs[0, 0].plot(time, quaternions)
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Quaternions')

    # 時間と誤差ｸｫｰﾀﾆｵﾝの関係
    axs[0, 1].plot(time, error_quaternions)
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Error Quaternions')

    # 時間とｵｲﾗｰ角の関係
    axs[1, 0].plot(time, euler_angles)
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Euler Angles')

    # 時間と回転軸の変化の関係
    # axs[1, 1].plot(time, theta_error_quaternions)
    axs[1, 1].plot(time, rotation_axes)
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Rotation Axes')

    # ｸﾞﾗﾌのﾚｲｱｳﾄを調整
    fig.tight_layout()

    # ｸﾞﾗﾌをPNGﾌｧｲﾙに保存
    fig.savefig('out/combined_graph.png')

def plot2(history, n_steps):

    time = np.linspace(0, n_steps/10, n_steps)
    quaternions = np.array(history)
    # error_quaternions = np.array(q_err_his)
    # # theta_error_quaternions = np.array(theta_err_his)
    # euler_angles = np.array(euler_angles_current_his)
    # rotation_axes = np.array(lambda_his)

    # ｸﾞﾗﾌの初期化
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 時間と外乱ﾄﾙｸの関係
    axs[0, 0].plot(time, quaternions)
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('disturbance_torque')

    # ｸﾞﾗﾌのﾚｲｱｳﾄを調整
    fig.tight_layout()

    # ｸﾞﾗﾌをPNGﾌｧｲﾙに保存
    fig.savefig('out/disturbance_torque_graph.png')