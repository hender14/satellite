import numpy as np
import transform as tf

# 簡易ｶﾙﾏﾝﾌｨﾙﾀｸﾗｽ
class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_estimate):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = initial_estimate

    def update(self, measurement):
        # 予測ｽﾃｯﾌﾟ
        prediction = self.estimate

        # 更新ｽﾃｯﾌﾟ
        kalman_gain = self.process_noise / (self.process_noise + self.measurement_noise)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.process_noise = (1 - kalman_gain) * self.process_noise

        return self.estimate


class DisturbTorq:
    def __init__(self):
        # ｶﾙﾏﾝﾌｨﾙﾀ初期化
        self.disturb_torque_filter = SimpleKalmanFilter(0.1, 0.1, np.zeros(3))
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