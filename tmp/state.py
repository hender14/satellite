import numpy as np
import matplotlib.pyplot as plt

def random_unit_quaternion():
    """
    ランダムな単位クォータニオンを生成します。
    """
    u1, u2, u3 = np.random.random(3)
    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([q0, q1, q2, q3])

# 3Dベクトル用のクロス積関数
def cross_product(v1, v2):
    return np.array([v1[1]*v2[2] - v1[2]*v2[1],
                     v1[2]*v2[0] - v1[0]*v2[2],
                     v1[0]*v2[1] - v1[1]*v2[0]])

# Quaternionを使用した回転
def rotate_vector(vector, quaternion):
    q_conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    vec_quat = np.concatenate(([0], vector))
    vec_rotated = quaternion_multiply(quaternion_multiply(quaternion, vec_quat), q_conjugate)
    return vec_rotated[1:]

# クォータニオンの積
def quaternion_multiply(q1, q2):
    return np.array([q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                     q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                     q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                     q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]])

# PID制御器クラス
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

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

# 時間、角度誤差、制御出力を保存するリスト
time_list = []
angle_error_list = []
control_output_list = []

# パラメータ
# パラメータ設定
kp = 0.015 #比例
ki = 0.6 #積分
# ki = 0.1 #積分
kd = 0. #微分
# kd = 0.05 #微分
dt = 0.1
max_iterations = 100

# 目標姿勢（クォータニオン）
target_quaternion = np.array([1, 0, 0, 0])  # 単位クォータニオン（回転なし）

# PID制御器の初期化
pid_controller = PIDController(kp, ki, kd, dt)

# # 初期姿勢（クォータニオン）
# current_quaternion = np.array([1, 0, 0, 0])

# ランダムな初期姿勢を生成
initial_quaternion = random_unit_quaternion()

# センサーデータ（ここではダミーデータ）
sensor_data = {
    "quaternion": initial_quaternion
}

for i in range(max_iterations):
    # 姿勢誤差計算
    current_quaternion = sensor_data["quaternion"]
    quaternion_error = quaternion_multiply(target_quaternion, current_quaternion)

    # クォータニオン誤差をオイラー角に変換
    angle_error = np.array([2 * np.arctan2(quaternion_error[1], quaternion_error[0]),
                            2 * np.arctan2(quaternion_error[2], quaternion_error[0]),
                            2 * np.arctan2(quaternion_error[3], quaternion_error[0])])


    # 姿勢制御
    control_output = pid_controller.update(angle_error)

    # 姿勢更新
    # ここでは制御出力を直接姿勢に反映していますが、実際のシステムでは制御出力はトルクに変換され、その後姿勢が更新されます。
    control_quaternion = np.concatenate(([1], control_output * dt / 2))
    current_quaternion = quaternion_multiply(control_quaternion, current_quaternion)

    # センサーデータの更新（ここではダミーデータ）
    sensor_data["quaternion"] = current_quaternion

    print(f"Time: {i*dt:.1f}s, Angle Error: {np.degrees(angle_error)}, Control Output: {control_output}")

    # データをリストに保存
    time_list.append(i * dt)
    angle_error_list.append(np.degrees(angle_error))
    control_output_list.append(control_output)

# 角度誤差のグラフ
plt.figure()
for i in range(3):
    plt.plot(time_list, np.array(angle_error_list)[:, i], label=f'Axis {i+1}', linestyle='-', marker='o', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Angle Error (degrees)')
plt.title('Angle Error vs Time')
plt.grid()
plt.legend()
plt.savefig('angle_error_vs_time.png', dpi=300)

# 制御出力のグラフ
plt.figure()
for i in range(3):
    plt.plot(time_list, np.array(control_output_list)[:, i], label=f'Axis {i+1}', linestyle='--', marker='o', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Control Output')
plt.title('Control Output vs Time')
plt.grid()
plt.legend()
plt.savefig('control_output_vs_time.png', dpi=300)

plt.show()