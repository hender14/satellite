import numpy as np

# PID制御器ｸﾗｽ
class PIDController:
    def __init__(self, kp, ki, kd, dt, arraynum):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = np.zeros(arraynum)
        self.prev_error = np.zeros(arraynum)

    def update(self, error):
        # 比例項
        proportional = self.kp * error

        # 積分項
        self.integral += error * self.dt
        integral_term = self.ki * self.integral

        # 微分項
        derivative = (error - self.prev_error) / self.dt
        derivative_term = self.kd * derivative

        # PID制御出力
        output = proportional + integral_term + derivative_term

        # 更新
        self.prev_error = error

        return output