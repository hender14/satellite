import numpy as np
from scipy.optimize import minimize

# 仮の衛星位置データ（単位：キロメートル）
observations = [
    np.array([-6144.333, 3491.773, 2578.643]),
    np.array([-5534.040, 4039.998, 3354.663]),
    np.array([-4762.633, 4474.107, 4024.830]),
    np.array([-3894.103, 4777.468, 4548.085]),
    np.array([-2980.972, 4935.636, 4883.337]),
    np.array([-2043.511, 4943.266, 4999.919]),
]

# それぞれの観測データに対応する時刻（単位：秒）
times = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
mu = 3.986004418e5  # Gravitational constant times Earth's mass (km^3/s^2)

# 観測された位置と計算された位置の差を計算する関数
def residuals(orbit_elements, observations, times, mu):
    calculated_positions = []
    for t in times:
        # 与えられた軌道要素を使用して、時刻tでの衛星の位置を計算する
        calculated_position = calculate_position(orbit_elements, t, mu)
        calculated_positions.append(calculated_position)

    # 観測された位置と計算された位置の差を計算する
    diff = np.array(observations) - np.array(calculated_positions)
    return diff.flatten()

# 軌道要素と時刻が与えられた場合の衛星の位置を計算する関数
def calculate_position(orbit_elements, time, mu):
    # 軌道要素を抽出する
    a, e, i, omega, w, M0 = orbit_elements

    # 与えられた時刻での平均近点離角を計算する
    n = np.sqrt(mu / a**3)
    M = M0 + n * (time - times[0])

    # 離心近点離角のケプラー方程式を解く
    E = solve_keplers_equation(M, e)

    # 真近点離角を計算する
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    # 軌道面内の衛星位置を計算する
    r = a * (1 - e * np.cos(E))
    position_orbital_plane = np.array([r * np.cos(nu), r * np.sin(nu), 0])

    # 位置ベクトルを地球中心慣性座標系に回転させる
    position_eci = rotate_to_eci(position_orbital_plane, i, omega, w)

    return position_eci

# ケプラー方程式を解く関数
def solve_keplers_equation(M, e):
    E = M
    for _ in range(10):
        E = M + e * np.sin(E)
    return E


# 軌道面から地球中心慣性座標系への位置ベクトルの回転を行う関数
def rotate_to_eci(position_orbital_plane, i, omega, w):
    R = np.array([
        [np.cos(omega) * np.cos(w) - np.sin(omega) * np.sin(w) * np.cos(i),
         -np.cos(omega) * np.sin(w) - np.sin(omega) * np.cos(w) * np.cos(i),
         np.sin(omega) * np.sin(i)],
        [np.sin(omega) * np.cos(w) + np.cos(omega) * np.sin(w) * np.cos(i),
         -np.sin(omega) * np.sin(w) + np.cos(omega) * np.cos(w) * np.cos(i),
         -np.cos(omega) * np.sin(i)],
        [np.sin(w) * np.sin(i),
         np.cos(w) * np.sin(i),
         np.cos(i)]
    ])
    position_eci = R @ position_orbital_plane
    return position_eci

# 軌道要素を最適化して、残差の二乗和を最小化する
def objective_function(orbit_elements, observations, times, mu):
    res = residuals(orbit_elements, observations, times, mu)
    return np.sum(res**2)

# 軌道要素の初期推定値
initial_guess = [7000, 0.01, np.deg2rad(98), np.deg2rad(50), np.deg2rad(90), np.deg2rad(0)]

# 最適化を実行する
result = minimize(objective_function, initial_guess, args=(observations, times, mu))

# 最適化された軌道要素を抽出する
optimized_orbit_elements = result.x
a, e, i, omega, w, M0 = optimized_orbit_elements

# 最適化された軌道要素を出力する
print("Optimized orbit elements:")
print("Semimajor axis (a):", a, "km")
print("Eccentricity (e):", e)
print("Inclination (i):", np.rad2deg(i), "degrees")
print("Right ascension of the ascending node (omega):", np.rad2deg(omega), "degrees")
print("Argument of periapsis (w):", np.rad2deg(w), "degrees")
print("Mean anomaly at epoch (M0):", np.rad2deg(M0), "degrees")

def calculate_position_velocity(orbit_elements, time, mu):
    # 上記の calculate_position 関数を使用して位置を計算
    position_eci = calculate_position(orbit_elements, time, mu)
    
    # 速度を計算するために、微小時間後の位置を計算
    delta_t = 1e-5
    position_eci_later = calculate_position(orbit_elements, time + delta_t, mu)
    
    # 速度を計算
    velocity_eci = (position_eci_later - position_eci) / delta_t
    
    # 2次元の位置情報
    position_2d = position_eci[:2]
    
    # 角度
    angle = np.arctan2(position_2d[1], position_2d[0])
    
    # 2次元の速度情報
    velocity_2d = velocity_eci[:2]
    
    # 角速度
    angular_velocity = (np.arctan2(velocity_2d[1], velocity_2d[0]) - angle) / delta_t
    
    return position_2d, angle, velocity_2d, angular_velocity

# 任意の時刻での2次元の位置情報、角度、速度、および角速度を計算
time = 100.0  # 任意の時刻（単位：秒）
position_2d, angle, velocity_2d, angular_velocity = calculate_position_velocity(optimized_orbit_elements, time, mu)

# 結果を表示
print("2D position:", position_2d, "km")
print("Angle:", np.rad2deg(angle), "degrees")
print("2D velocity:", velocity_2d, "km/s")
print("Angular velocity:", np.rad2deg(angular_velocity), "degrees/s")