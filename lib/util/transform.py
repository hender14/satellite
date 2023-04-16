import numpy as np
import math

def euler_to_rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    C = np.array([[ctheta * cpsi, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
                  [ctheta * spsi, sphi * stheta * spsi + cphi * cpsi, cphi * stheta * spsi - sphi * cpsi],
                  [-stheta, sphi * ctheta, cphi * ctheta]]).T
    return C

def rotation_matrix_to_quaternion(C):
    q4 = 0.5 * np.sqrt(1 + C[0, 0] + C[1, 1] + C[2, 2])
    q1 = 0.25 * (C[1, 2] - C[2, 1]) * q4 #本来はq4で割る
    q2 = 0.25 * (C[2, 0] - C[0, 2]) * q4
    q3 = 0.25 * (C[0, 1] - C[1, 0]) * q4
    return np.array([q1, q2, q3, q4])

def euler_to_quaternion(phi, theta, psi):
    C = euler_to_rotation_matrix(phi, theta, psi)
    # print("C = ", C)
    q = rotation_matrix_to_quaternion(C)
    # print("q = ", q)
    return q

def quaternion_to_euler(q):
    C = quaternion_to_rotation_matrix(q)
    euler_angle = rotation_matrix_to_euler(C)
    return euler_angle

def quaternion_to_euler_deg(q):
    C = quaternion_to_rotation_matrix(q)
    euler_angle = rotation_matrix_to_euler_deg(C)
    return euler_angle

def quaternion_to_axis_angle(q):
    C = quaternion_to_rotation_matrix(q)
    # print(f"C = {C}")
    lambdai, theta_err = rotation_matrix_to_axis_angle(C)
    return lambdai, theta_err

def quaternion_to_rotation_matrix(q):
    # Convert quaternion to rotation matrix
    q1, q2, q3, q4 = q
    C = np.array([
        [q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4)],
        [2 * (q1 * q2 - q3 * q4), -q1 ** 2 + q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q2 * q3 + q1 * q4)],
        [2 * (q1 * q3 + q2 * q4), 2 * (q2 * q3 - q1 * q4), -q1 ** 2 - q2 ** 2 + q3 ** 2 + q4 ** 2]
    ])
    return C

def rotation_matrix_to_euler(C):
    phi = np.arctan2(C[1, 2], C[2, 2])
    theta = np.arctan2(-C[0, 2], np.sqrt(C[1, 2]**2 + C[2, 2]**2))
    psi = np.arctan2(C[0, 1], C[0, 0])

    return np.array([phi, theta, psi])

def rotation_matrix_to_euler_deg(C):
    phi = np.arctan2(C[1, 2], C[2, 2])
    theta = np.arctan2(-C[0, 2], np.sqrt(C[1, 2]**2 + C[2, 2]**2))
    psi = np.arctan2(C[0, 1], C[0, 0])

    return np.array([math.degrees(phi), math.degrees(theta), math.degrees(psi)])

def rotation_matrix_to_axis_angle(C):
    # Compute the rotation angle
    theta = np.arccos(0.5 * ((C[0, 0] + C[1, 1] + C[2, 2]) - 1))
    
    # Compute the axis direction
    sin_theta = np.sin(theta)
    lambda_1 = (C[1, 2] - C[2, 1]) / (2 * sin_theta)
    lambda_2 = (C[2, 0] - C[0, 2]) / (2 * sin_theta)
    lambda_3 = (C[0, 1] - C[1, 0]) / (2 * sin_theta)
    lambdai = np.array([lambda_1, lambda_2, lambda_3])

    return lambdai, theta

def quaternion_multiply(q, p):
    # Multiply two quaternions
    q1, q2, q3, q4 = q
    p1, p2, p3, p4 = p
    # print("p:{}, q:{}".format(p, q))

    r1 = q4 * p1 + q1 * p4 + q3 * p2 - q2 * p3
    r2 = q4 * p2 + q1 * p3 + q2 * p4 - q3 * p1
    r3 = q4 * p3 - q1 * p2 + q2 * p1 + q3 * p4
    r4 = q4 * p4 - q1 * p1 - q2 * p2 - q3 * p3
    # print(f"pq = {np.array([r1, r2, r3, r4])}")

    return np.array([r1, r2, r3, r4])

def quaternion_inverse(q):
    # Calculate the inverse of a quaternion
    q_star = q.copy()
    q_star[:3] = -q_star[:3]
    return q_star #後で戻す
    # return q_star / np.linalg.norm(q)

def tmp_multi(w, tmp):
    w1, w2, w3 = w
    tmp1, tmp2, tmp3 = tmp

    tmp1 = w2*tmp3 - w3*tmp2
    tmp2 = w3*tmp1 - w1*tmp3
    tmp3 = w1*tmp2 - w2*tmp1

    return np.array([tmp1, tmp2, tmp3])