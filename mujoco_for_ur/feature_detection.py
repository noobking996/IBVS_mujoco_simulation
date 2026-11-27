"""《具身智能控制基础》任务 A：从 RGB 图像提取四个目标角点。"""

from itertools import permutations

import cv2 as cv
import numpy as np

# ---- 检测参数 ---------------------------------------------------------
#--颜色分割--#
_HUE_RED_LOW = (0, 12)          # 低色相段的 H 范围
_HUE_RED_HIGH = (160, 180)      # 高色相段的 H 范围
_SAT_MIN = 50                   # 饱和度下限，滤掉灰白背景
_VAL_MIN = 40                   # 亮度下限，滤掉近黑像素
_MORPH_KERNEL_SIZE = 3          # (去噪 + 补洞)形态学核尺寸
#--轮廓筛选--#
_MIN_CONTOUR_AREA = 30.0        # 单位: px^2，小于该面积视为噪点
_EDGE_MARGIN = 5                # 单位: px，距图像边界不足该值视为贴边或被裁切
#--透视四边形--#
_APPROX_EPS_START = 0.02        # (多边形逼近)初始允许误差(占周长比例)
_APPROX_EPS_STEP = 0.005        # 允许误差自增步长
_APPROX_EPS_MAX = 0.20          # 允许误差上限
_SUBPIX_MAX_SHIFT = 1.0         # px，亚像素精化允许的最大偏移，超出即回退
_SUBPIX_MIN_AREA_RATIO = 0.85   # 精化后面积至少保留原面积的比例，低于即回退
_SUBPIX_WIN_SIZE = (5, 5)        # 亚像素精化搜索窗口边长
_SUBPIX_ZERO_ZONE = (-1, -1)     # 零点区域, 默认(-1, -1): 由 cornerSubPix 自行计算
#--跨帧对应--#
_RAISE_ERROR = True             # 上帧角点格式有误时, 是否抛出错误


def _segment_red(image: np.ndarray) -> np.ndarray:
    """图像二值化, 提取红色区域掩模"""
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    lower_a = np.array([_HUE_RED_LOW[0], _SAT_MIN, _VAL_MIN], np.uint8)
    upper_a = np.array([_HUE_RED_LOW[1], 255, 255], np.uint8)
    lower_b = np.array([_HUE_RED_HIGH[0], _SAT_MIN, _VAL_MIN], np.uint8)
    upper_b = np.array([_HUE_RED_HIGH[1], 255, 255], np.uint8)
    # 红色区域筛选;(色相段取并集)
    mask = cv.bitwise_or(cv.inRange(hsv, lower_a, upper_a), # 低色相段红色
                         cv.inRange(hsv, lower_b, upper_b)) # 高色相段红色
    kernel = np.ones((_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE), np.uint8)
    # 去噪 + 补洞
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)     # 形态学开运算, 去噪
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)    # 形态学闭运算, 补洞
    return mask


def _select_target(mask: np.ndarray) -> np.ndarray | None:
    """从二值图像中提取并返回面积最大的合格轮廓; 无合格轮廓时返回 None。"""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   # 二值图像中找到(可能)多个轮廓点集
    candidates = [c for c in contours if cv.contourArea(c) >= _MIN_CONTOUR_AREA]    # 依据面积筛选轮廓, 得到筛选后的点集
    if not candidates:
        return None
    return max(candidates, key=cv.contourArea)  # 返回面积最大的一个轮廓点集


def _is_clipped(contour: np.ndarray, width: int, height: int) -> bool:
    """判断轮廓点集是否贴边/裁切。"""
    x, y, w, h = cv.boundingRect(contour)   # 外接矩形
    return (x < _EDGE_MARGIN or y < _EDGE_MARGIN
            or x + w > width - _EDGE_MARGIN
            or y + h > height - _EDGE_MARGIN)


def _extract_quad(contour) -> np.ndarray | None:
    """从轮廓点集中提取 4 个透视顶点"""
    # 凸包轮廓点集构建
    hull = cv.convexHull(contour)
    perimeter = cv.arcLength(hull, True)
    if perimeter <= 0:
        return None
    #  四边形顶点逼近
    eps = _APPROX_EPS_START
    while eps <= _APPROX_EPS_MAX:
        approx:np.ndarray = cv.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        eps += _APPROX_EPS_STEP
    return None


def _refine_subpix(image: np.ndarray, quad: np.ndarray):
    """四角点坐标的有条件亚像素精化, 返回精化后的点集或原点集"""
    # 灰度图 + 迭代终止条件(最大 30 次 / 精度 0.01)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    # 创建副本, 用于精化
    refined = quad.reshape(-1, 1, 2).copy()
    # 亚像素精化
    try:
        cv.cornerSubPix(gray, refined, _SUBPIX_WIN_SIZE, _SUBPIX_ZERO_ZONE,
                        criteria)
    except cv.error:                        # 抛异常, 回退原四角点
        return quad
    refined = refined.reshape(4, 2)
    if not np.all(np.isfinite(refined)):    # 含 NaN / inf, 回退原四角点
        return quad
    # 发散判断: 任一角点位移超过阈值
    displacements: np.ndarray = np.linalg.norm(refined - quad, axis=1)  # 精化后四角点位移
    if displacements.max() > _SUBPIX_MAX_SHIFT:
        return quad
    # 面积异常判断: 若精化后面积不足原面积的给定比例, 回退
    area_before = cv.contourArea(quad.reshape(-1, 1, 2))
    area_after = cv.contourArea(refined.reshape(-1, 1, 2))
    if area_before > 0 and area_after < _SUBPIX_MIN_AREA_RATIO * area_before:
        return quad
    return refined                          # 以上判据均未触发, 采用精化结果


def _order_clockwise(points: np.ndarray) -> np.ndarray:
    """角点重排序(绝对位置): 按质心极角顺时针排序, 以左上方角点作 1 号"""
    center = points.mean(axis=0)    # 几何中心, 作为极坐标原点
    pole_angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])    # 各点极角
    ordered_pts = points[np.argsort(pole_angles)]   # 顺时针排序后的数组, 形状:(N, 2)
    No1_index = int(np.argmin(ordered_pts.sum(axis=1))) # 1号点在ordered_pts中的索引
    return np.roll(ordered_pts, -No1_index, axis=0) # 返回重排序(N, 2)形状点集, 以1号点为首个元素


def _order_relative(points: np.ndarray, previous) -> np.ndarray | None:
    """角点重排序(相对位置): 依据与上帧角点相对位置排序, 保持物理角点编号不随图像方位跳变"""
    if previous is None:    #首次, 直接跳过
        return None
    prev = np.asarray(previous, dtype=np.float64)
    if prev.shape != (4, 2) or not np.all(np.isfinite(prev)):
        if _RAISE_ERROR:
            raise ValueError("FATAL: Invalid format of previous_points detected !!!")
        return None
    best_perm, best_cost = None, np.inf
    for perm in permutations(range(4)): # 遍历四角点顺序排列(permutations)
        cost = float(np.sum((points[list(perm)] - prev) ** 2))  # 计算当前排列与上帧角点距离平方和
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return points[list(best_perm)]


def detect_feature_points(image, previous_points=None):
    """返回 (points, success)，由学生实现。

    输入：
        image: uint8 RGB 图像，形状 (height, width, 3)，不是 BGR。
        previous_points: 上次成功检测的 (4, 2) 像素坐标，首次为 None。
            此信息仅用于匹配角点身份，不允许用旧坐标替代本帧检测。
    输出：
        成功：(浮点 ndarray，形状 (4, 2)，True)，每行为 (u, v)。
        失败：(None, False)，包括无目标、不完整、贴边或被裁切的情况。
    约定：
        初次按左上、右上、右下、左下排列；后续保持同一物理角点的编号。
        不修改输入数组，不读取 MuJoCo 位姿或用几何投影代替图像检测。

    TODO A：颜色分割、轮廓筛选、透视四角点提取、精化与跨帧对应。
    提示：红色色相跨 HSV 两端；倾斜四边形不能用外接矩形代替。
    建议拒绝面积过小（例如 <30 px²）或距边界不足 5 px 的目标。
    """
    if image is None:
        return None, False
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        return None, False
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    height, width = frame.shape[:2]

    # 1. 颜色分割：红色双区间取并集，再开运算去噪、闭运算补洞
    mask = _segment_red(frame)

    # 2. 轮廓筛选：面积过小视作噪点，贴边或被裁切直接判失败
    target = _select_target(mask)
    if target is None or _is_clipped(target, width, height):
        return None, False

    # 3. 透视四边形：凸包 + approxPolyDP 逼近四顶点，再做亚像素精化
    quad = _extract_quad(target)
    if quad is None:
        return None, False
    quad = _refine_subpix(frame, quad)
    points = quad.astype(np.float64)

    # 4. 角点编号：首帧按左上/右上/右下/左下顺时针，后续帧按上一帧最近匹配
    matched_pts = None
    if previous_points is not None:
        matched_pts = _order_relative(points, previous_points)
    points = _order_clockwise(points) if matched_pts is None else matched_pts
    return points, True
