import numpy as np
import scipy.integrate as spi
from scipy.spatial.transform import Rotation as R
from .util import transform as tf, pid

class AngulVelocity:
    def __init__(self, omega_current, tgt_omega):
        # ｶﾙﾏﾝﾌｨﾙﾀ初期化
        self.pid_controller = pid.PIDController(0.0, 0., 0., 0.1) # kp, ki, kd, dt
        self.omega_current = omega_current  # 初期角速度
        self.tgt_goal = tgt_omega  # 目標角速度
        # Constraints
        self.tgt_omega = tgt_omega  # 目標角速度
        self.max_angular_rate_change = 0.01  # [rad/s]: 最大加速ﾚｰﾄ
        self.angular_rate_change = 0.01  # [rad/s]: 最大加速ﾚｰﾄ
        self.omega_pid = omega_current
        self.omega_inst = tgt_omega
        self.omega_tgt = tgt_omega
        self.omega_err = tgt_omega

    def calc_omega(self, I, omega, u, I_inv, dt):
        self.calc_inst_omega(omega)
        # # Solve the motion equations
        tmp_l = (I @ omega.T).T
        tmp_r = u - tf.tmp_multi(omega, tmp_l)
        omega_dot = (I_inv @ tmp_r.T).T
        omega_dot_limit = self.constrain_angular_rate(omega_dot)

        self.omega_current += omega_dot_limit * dt
        self.constrain_angular(self.omega_current)

        self.omega_err = self.omega_tgt - self.omega_current
        # print("omega_tgt:{} omega_current:{}".format(self.omega_tgt, self.omega_current))
        # print("omega:{} {}".format(self.omega_current, self.omega_err))
        self.pid_control()

        return self.omega_current
    
    def pid_control(self):
        # omega_err = tf.quaternion_to_euler(q_err)
        self.omega_pid = self.pid_controller.update(self.omega_err)
    
    def calc_inst_omega(self, omega):
        # err = np.linalg.norm(self.omega_pid - omega)
        # if abs(err) < self.angular_rate_change:
        #     self.omega_inst = self.omega_pid
        # else :
        #     self.omega_inst = abs(omega) + self.angular_rate_change
        self.omega_inst = self.omega_pid + self.angular_rate_change
        
    # Functions to apply angular rate constraints
    def constrain_angular_rate(self, delta_angular_rate):
        return np.clip(delta_angular_rate, -self.max_angular_rate_change, self.max_angular_rate_change)

    # Functions to apply angular constraints
    def constrain_angular(self, delta_angular):
        self.omega_current = np.clip(delta_angular, -self.omega_inst, self.omega_inst)