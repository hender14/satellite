import numpy as np
from .util import transform as tf

class Quaternion:
    def __init__(self, q_current, q_tgt):
        self.q_current = q_current  # 初期角速度
        self.q_tgt = q_tgt

    def calc_quater(self, omega_current, dt):
        # ｸｫｰﾀﾆｵﾝと角速度を更新
        q_dot = 0.5 * tf.quaternion_multiply(self.q_current, np.append(omega_current, 0))
        self.q_current += q_dot * dt

        q_err = tf.quaternion_multiply(tf.quaternion_inverse(self.q_current), self.q_tgt)

        return self.q_current, q_err