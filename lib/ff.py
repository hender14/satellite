import numpy as np
from .util import transform as tf, kfilter as kf

class DisturbTorq:
    def __init__(self):
        # ｶﾙﾏﾝﾌｨﾙﾀ初期化
        self.disturb_torque_filter = kf.SimpleKalmanFilter(0.1, 0.1, np.zeros(3))
        self.alpha, self.beta = 10., 50.

        self.period = 100 # 外乱ﾄﾙｸの周期
        self.amplitude = np.array([0.5, 0.5, 0.5]) # 外乱ﾄﾙｸの振幅

    # 外乱ﾄﾙｸ生成用関数
    def generate_disturbance_torque(self, step):
        return self.amplitude * np.sin(2 * np.pi * step / self.period) + np.random.normal(0, 0.5, 3)

    def calc_disturb_trq(self, q_err, omega_err, step):

        # 外乱ﾄﾙｸ生成
        disturbance_torque = self.generate_disturbance_torque(step)

        # 外乱ﾄﾙｸ推定
        estimated_disturbance_torque = self.disturb_torque_filter.update(disturbance_torque)

        state_fb = self.alpha * q_err[:3] + self.beta * omega_err

        # Solve the motion equations
        u = state_fb + estimated_disturbance_torque

        return u