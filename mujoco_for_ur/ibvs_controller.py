"""《具身智能控制基础》任务 B：四点 IBVS 交互矩阵与速度控制律。"""

import numpy as np


def calculate_interaction_matrix(points, depths, intrinsics) -> np.ndarray:
    """计算归一化图像坐标的交互矩阵 L，形状 (8, 6)。

    points: (4, 2)，四点像素坐标，每行 (u, v)。
    depths: (4,)，与各点一一对应的光学 Z 深度，单位 m，必须有限且 >0。
    intrinsics: (fx, fy, cx, cy)，单位 px。
    输出行顺序：[x1, y1, x2, y2, x3, y3, x4, y4] 的时间导数。
    输出列顺序：相机速度 [vx, vy, vz, wx, wy, wz]。
    相机光学坐标轴：向右、向下、向前；线速度 m/s，角速度 rad/s。

    TODO B1：从针孔模型推导并实现逐点交互矩阵；拒绝无效深度。
    注意：输入是像素坐标，输出描述归一化坐标的变化率。
    """
    points = np.asarray(points, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if points.shape != (4, 2):
        raise ValueError("!! points 形状不为 (4, 2) !!")
    if depths.shape != (4,):
        raise ValueError("!! depths 形状不为 (4,) !!")
    # 深度为非正或非有限时无法定义 1/Z, 直接拒绝
    if not np.all(np.isfinite(depths)) or np.any(depths <= 0):
        raise ValueError("特征深度必须为有限正数, 单位 m")
    if intrinsics.shape != (4,):
        raise ValueError("intrinsics 必须为 (fx, fy, cx, cy) 四个数")
    fx, fy, cx, cy = intrinsics
    if not np.all(np.isfinite(intrinsics)) or fx <= 0 or fy <= 0:
        raise ValueError("intrinsics 必须为有限正数, 单位 px")

    # 归一化图像坐标: x = (u - cx) / fx, y = (v - cy) / fy
    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy
    z = depths

    # 逐点 2x6 块; 行序 [x_dot, y_dot], 列序 [vx, vy, vz, wx, wy, wz]
    interaction = np.zeros((8, 6), dtype=np.float64)
    # x_dot 行
    interaction[0::2, 0] = -1.0 / z             # vx
    pass                                        # vy
    interaction[0::2, 2] = x / z                # vz
    interaction[0::2, 3] = x * y                # wx
    interaction[0::2, 4] = -(1.0 + x ** 2)      # wy
    interaction[0::2, 5] = y                    # wz
    # y_dot 行
    pass                                        # vx
    interaction[1::2, 1] = -1.0 / z             # vy
    interaction[1::2, 2] = y / z                # vz
    interaction[1::2, 3] = 1.0 + y ** 2         # wx
    interaction[1::2, 4] = -x * y               # wy
    interaction[1::2, 5] = -x                   # wz
    return interaction


def compute_joint_velocity(points, desired_points, depths, intrinsics,
                           robot_jacobian, lambda_gain=2.0, damping=1e-4) -> np.ndarray:
    """计算关节速度 qdot，返回形状 (n,) 的有限浮点 ndarray，单位 rad/s。

    points / desired_points: (4, 2)，当前/期望像素坐标，角点身份一致。
    depths: (4,)，当前逐点光学 Z 深度，单位 m。
    intrinsics: (fx, fy, cx, cy)。
    robot_jacobian: (6, n)，把 qdot 映射到相机光学坐标系的 [v, omega]。
    lambda_gain: 正数控制增益，期望误差随时间衰减。
    damping: 阻尼参数，用于处理病态矩阵。

    TODO B2：
        1. 构造归一化图像误差（约定 current - desired）。
        2. 调用 B1，结合机器人雅可比建立图像误差与关节速度的关系。
        3. 设计负反馈，用阻尼最小二乘求解关节速度。
    不读取目标真值位姿，不将误差直接当作关节角/关节位置。
    速度限幅、丢失目标停车和收敛停车由外层框架提供。
    """
    points = np.asarray(points, dtype=np.float64)
    desired = np.asarray(desired_points, dtype=np.float64)
    jacobian = np.asarray(robot_jacobian, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if points.shape != (4, 2) or desired.shape != (4, 2):
        raise ValueError("points 与 desired_points 都必须是 (4, 2) 角点数组")
    if jacobian.ndim != 2 or jacobian.shape[0] != 6:
        raise ValueError("robot_jacobian 形状必须为 (6, n), 行对应 [v, omega]")
    if intrinsics.shape != (4,):
        raise ValueError("intrinsics 必须为 (fx, fy, cx, cy) 四个数")
    if lambda_gain <= 0:
        raise ValueError("lambda_gain 必须为正数, 否则误差不衰减")
    if damping < 0:
        raise ValueError("damping 不能为负")
    fx, fy = float(intrinsics[0]), float(intrinsics[1])

    # 归一化图像误差 e = s - s*, 行序与 B1 一致
    error = np.empty(8, dtype=np.float64)
    error[0::2] = (points[:, 0] - desired[:, 0]) / fx
    error[1::2] = (points[:, 1] - desired[:, 1]) / fy

    # 图像变化率 s_dot = L @ J @ q_dot, 取负反馈 q_dot = -lambda (A^T A + mu^2 I)^-1 A^T e
    interaction = calculate_interaction_matrix(points, depths, intrinsics)
    A = interaction @ jacobian              # 图像雅可比矩阵 (8, 6)
    # (正则化)格拉姆矩阵 A^T A + mu^2 (mu = damping)
    n = jacobian.shape[1]                   # 关节数: 6
    gram = A.T @ A + damping ** 2 * np.eye(n)
    q_dot = -lambda_gain * np.linalg.solve(gram, A.T @ error)   # 单位: rad/s
    return q_dot
