import numpy as np
import scipy.integrate as spi
from scipy.spatial.transform import Rotation as R
from .util import transform as tf, pid, kfilter as kf

class AngulVelocity:
    def __init__(self, omega_current, tgt_omega):
        self.pid_controller = pid.PIDController(0.0, 0., 0., 0.1, 3) # kp, ki, kd, dt, arraynum
        self.sensor_filter = kf.SimpleKalmanFilter(0.1, 0.1, omega_current) # ｶﾙﾏﾝﾌｨﾙﾀ初期化
        self.omega_current = omega_current  # 初期角速度
        self.max_angular_rate_change = 0.01  # [rad/s]: 最大加速ﾚｰﾄ
        self.angular_rate_change = 0.003  # [rad/s]: 加速ﾚｰﾄ
        self.omega_pid = omega_current #PID制御器の出力
        self.omega_inst = tgt_omega #指示角速度
        self.omega_tgt = tgt_omega #目標角速度
        self.omega_err = tgt_omega #目標角速度と現在角速度の差

    def calc_omega(self, I, omega, u, I_inv, dt):
        # calc kalman filter
        self.omega_current = self.sensor_filter.update(omega)

        # calc instruction omega
        self.calc_inst_omega()

        # calc dw/dt
        tmp_l = (I @ omega.T).T
        tmp_r = u - tf.tmp_multi(omega, tmp_l)
        omega_dot = (I_inv @ tmp_r.T).T

        # limit dw/dt
        omega_dot_limit = self.constrain_angular_rate(omega_dot)

        # calc w
        self.omega_current += omega_dot_limit * dt

        # limit w
        self.constrain_angular()

        # calc omega err
        self.omega_err = self.omega_tgt - self.omega_current

        # calc omega pid
        self.pid_control()

        return self.omega_current
    
    
    def calc_inst_omega(self):
        err = np.linalg.norm(self.omega_pid - self.omega_current)
        if abs(err) < self.angular_rate_change:
            self.omega_inst = self.omega_pid
        else :
            self.omega_inst = abs(self.omega_current) + self.angular_rate_change
        
    def pid_control(self):
        self.omega_pid = self.pid_controller.update(self.omega_err)

    # Functions to apply angular rate constraints
    def constrain_angular_rate(self, delta_angular_rate):
        return np.clip(delta_angular_rate, -self.max_angular_rate_change, self.max_angular_rate_change)

    # Functions to apply angular constraints
    def constrain_angular(self):
        self.omega_current = np.clip(self.omega_current, -self.omega_inst, self.omega_inst)