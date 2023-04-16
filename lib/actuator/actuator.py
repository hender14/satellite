import numpy as np

class Actuator:
    def __init__(self, omega_current):
        self.torque = omega_current
        self.trans = 0.1 # ｱｸﾁｭｴｰﾀ出力の変化率

    # ｱｸﾁｭｴｰﾀへ出力命令を実施
    def output(self, omega):
        self.calc_output(omega)
    
    def calc_output(self, omega):
        self.torque = omega * self.trans