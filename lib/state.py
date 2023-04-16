import numpy as np
from .util import transform as tf, pid, kfilter as kf

class Quaternion:
    def __init__(self, q_current, q_tgt, euler_current, euler_tgt):
        self.pid_controller = pid.PIDController(0.0, 0., 0., 0.1, 4) # kp, ki, kd, dt, arraynum
        self.sensor_filter = kf.SimpleKalmanFilter(0.1, 0.1, q_current) # ｶﾙﾏﾝﾌｨﾙﾀ初期化
        self.q_current = q_current  # 初期姿勢
        self.q_tgt = q_tgt  # 目標姿勢
        self.angular_rate_change = 0.003  # [rad/s]: 速度ﾚｰﾄ
        # self.euler_tgt = euler_tgt
        self.q_pid = q_current #PID制御器の出力
        self.q_inst = q_tgt #指示姿勢
        self.q_current = q_current #現在姿勢
        self.q_err = q_tgt #目標姿勢と現在姿勢の差

    # def calc_quater(self, q_current, omega_current, dt):
    def calc_quater(self, sensor_data, omega_current, dt):

        # calc kalman filter
        self.q_current = self.sensor_filter.update(sensor_data)

        # calc instruction euler
        self.calc_inst_euler()

        # calc dq/dt, q
        q_dot = 0.5 * tf.quaternion_multiply(self.q_current, np.append(omega_current, 0))
        self.q_current += q_dot * dt

        # limit q
        self.constrain_angular()

        # calc omega err
        q_err = tf.quaternion_multiply(tf.quaternion_inverse(self.q_current), self.q_tgt)

        # calc omega pid
        self.pid_control()

        return self.q_current, q_err
    
    def calc_inst_euler(self):
        err = np.linalg.norm(self.q_pid - self.q_current)
        if abs(err) < self.angular_rate_change:
            self.q_inst = self.q_pid
        else :
            self.q_inst = abs(self.q_current) + self.angular_rate_change

    
    def pid_control(self):
        self.q_pid = self.pid_controller.update(self.q_err)

    # Functions to apply angular constraints
    def constrain_angular(self):
        self.q_current = np.clip(self.q_current, -self.q_inst, self.q_inst)