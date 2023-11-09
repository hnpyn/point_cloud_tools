import math

import numpy as np


def Rotate_Matrix(point, xr, yr, zr):
    _matrix_x = np.zeros((3, 3))
    _matrix_y = np.zeros((3, 3))
    _matrix_z = np.zeros((3, 3))
    _rotate_x = (xr / 180) * math.pi
    _rotate_y = (yr / 180) * math.pi
    _rotate_z = (zr / 180) * math.pi
    _matrix_x[0][0] = 1
    _matrix_x[1][1] = math.cos(_rotate_x)
    _matrix_x[1][2] = math.sin(_rotate_x)
    _matrix_x[2][1] = -math.sin(_rotate_x)
    _matrix_x[2][2] = math.cos(_rotate_x)
    _matrix_y[1][1] = 1
    _matrix_y[0][0] = math.cos(_rotate_y)
    _matrix_y[0][2] = -math.sin(_rotate_y)
    _matrix_y[2][0] = math.sin(_rotate_y)
    _matrix_y[2][2] = math.cos(_rotate_y)
    _matrix_z[2][2] = 1
    _matrix_z[0][0] = math.cos(_rotate_z)
    _matrix_z[0][1] = math.sin(_rotate_z)
    _matrix_z[1][0] = -math.sin(_rotate_z)
    _matrix_z[1][1] = math.cos(_rotate_z)
    point = np.dot(point, _matrix_x)
    point = np.dot(point, _matrix_y)
    point = np.dot(point, _matrix_z)
    return point


if __name__ == "__main__":
    cur_velodyne = np.fromfile(
        "/home/n/Workspace/data/tmp/debug_pose/1/1691654341850207.bin", dtype=np.float32
    ).reshape(-1, 6)
    next_velodyne = np.fromfile(
        "/home/n/Workspace/data/tmp/debug_pose/1/1691654342450127.bin", dtype=np.float32
    ).reshape(-1, 6)
    x, y, z, roll, pitch, yaw = (
        399.56148040315134,
        1066.6322188609104,
        1.5612177039165946,
        0.0172382835,
        -0.00137157587,
        2.12555575,
    )
    x_n, y_n, z_n, roll_n, pitch_n, yaw_n = (
        397.99606673494924,
        1069.1173080862802,
        1.5646509888062305,
        0.0148270065,
        0.00124352169,
        2.12873077,
    )
    cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], yaw / math.pi * 180, 0, 0)
    cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], 0, pitch / math.pi * 180, 0)
    cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], 0, 0, roll / math.pi * 180)
    cur_velodyne[:, 0] += x
    cur_velodyne[:, 1] += y
    cur_velodyne[:, 2] += z

    cur_velodyne[:, 0] -= x_n
    cur_velodyne[:, 1] -= y_n
    cur_velodyne[:, 2] -= z_n
    # cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], -pose_patch[0][6] / math.pi * 180, 0, 0)
    # cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], 0, -pose_patch[0][5] / math.pi * 180, 0)
    # cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], 0, 0, -pose_patch[0][4] / math.pi * 180)

    cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], 0, 0, -roll_n / math.pi * 180)
    cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], 0, -pitch_n / math.pi * 180, 0)
    cur_velodyne[:, :3] = Rotate_Matrix(cur_velodyne[:, :3], -yaw_n / math.pi * 180, 0, 0)

    points = np.concatenate([cur_velodyne, next_velodyne], axis=0)
    points.astype(np.float32).tofile("data/bin/debug_pose.bin")
