import numpy as np
import transform as tf

class AngulVelocity:
    def __init__(self, omega_current):
        self.omega_current = omega_current  # 初期角速度

    def calc_omega(self, I, omega, u, I_inv, dt):
        # # Solve the motion equations
        tmp_l = (I @ omega.T).T
        tmp_r = u - tf.tmp_multi(omega, tmp_l)
        omega_dot = (I_inv @ tmp_r.T).T
        self.omega_current += omega_dot * dt

        return self.omega_current