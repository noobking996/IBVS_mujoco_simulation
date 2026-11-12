"""《具身智能控制基础》任务 A：从 RGB 图像提取四个目标角点。"""

from itertools import permutations

import cv2 as cv
import numpy as np

# ---- 检测参数 ---------------------------------------------------------
# 红色在 HSV 色相环上跨 0°/180° 两端，因此需要两段区间取并集
_HUE_RED_LOW = (0, 12)          # 低色相段的 H 范围
_HUE_RED_HIGH = (160, 180)      # 高色相段的 H 范围
_SAT_MIN = 50                   # 饱和度下限，滤掉灰白背景
_VAL_MIN = 40                   # 亮度下限，滤掉近黑像素
_MIN_CONTOUR_AREA = 30.0        # px^2，小于该面积视为噪点
_EDGE_MARGIN = 5                # px，距图像边界不足该值视为贴边或被裁切
_MORPH_KERNEL = 3               # 形态学核尺寸
_APPROX_EPS_START = 0.02        # 多边形逼近的初始精度（占周长比例）
_APPROX_EPS_STEP = 0.005        # 精度逐步放宽的步长
_APPROX_EPS_MAX = 0.20          # 精度放宽上限
_SUBPIX_MAX_SHIFT = 1.0         # px，亚像素精化允许的最大偏移，超出即回退


def _segment_red(image):
    """RGB -> HSV 双区间红色分割，形态学开闭运算去噪补洞，返回 uint8 掩膜。"""
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    lower_a = np.array([_HUE_RED_LOW[0], _SAT_MIN, _VAL_MIN], np.uint8)
    upper_a = np.array([_HUE_RED_LOW[1], 255, 255], np.uint8)
    lower_b = np.array([_HUE_RED_HIGH[0], _SAT_MIN, _VAL_MIN], np.uint8)
    upper_b = np.array([_HUE_RED_HIGH[1], 255, 255], np.uint8)
    mask = cv.bitwise_or(cv.inRange(hsv, lower_a, upper_a),
                         cv.inRange(hsv, lower_b, upper_b))
    kernel = np.ones((_MORPH_KERNEL, _MORPH_KERNEL), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask


def _select_target(mask):
    """在掩膜中挑选面积最大的合格轮廓；无合格候选时返回 None。"""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv.contourArea(c) >= _MIN_CONTOUR_AREA]
    if not candidates:
        return None
    return max(candidates, key=cv.contourArea)


def _is_clipped(contour, width, height):
    """目标外接框是否贴边或被裁切。"""
    x, y, w, h = cv.boundingRect(contour)
    return (x < _EDGE_MARGIN or y < _EDGE_MARGIN
            or x + w > width - _EDGE_MARGIN
            or y + h > height - _EDGE_MARGIN)


def _extract_quad(contour):
    """凸包 + 逐步放宽 epsilon 的 approxPolyDP，得到 4 个透视顶点。"""
    hull = cv.convexHull(contour)
    perimeter = cv.arcLength(hull, True)
    if perimeter <= 0:
        return None
    eps = _APPROX_EPS_START
    while eps <= _APPROX_EPS_MAX:
        approx = cv.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        eps += _APPROX_EPS_STEP
    return None


def _refine_subpix(image, quad):
    """cornerSubPix 亚像素精化；发散、非有限或面积异常时回退原四角点。"""
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    refined = quad.reshape(-1, 1, 2).copy()
    try:
        cv.cornerSubPix(gray, refined, (5, 5), (-1, -1), criteria)
    except cv.error:
        return quad
    refined = refined.reshape(4, 2)
    if not np.all(np.isfinite(refined)):
        return quad
    if np.linalg.norm(refined - quad, axis=1).max() > _SUBPIX_MAX_SHIFT:
        return quad
    area_before = cv.contourArea(quad.reshape(-1, 1, 2))
    area_after = cv.contourArea(refined.reshape(-1, 1, 2))
    if area_before > 0 and area_after < 0.85 * area_before:
        return quad
    return refined


def _order_clockwise(points):
    """按质心极角顺时针排序，并以 x+y 最小者（左上）作为 1 号角点。"""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def _match_previous(points, previous):
    """在 4! 全排列中取与上一帧距离平方和最小的编号，保持物理角点身份。"""
    prev = np.asarray(previous, dtype=np.float64)
    if prev.shape != (4, 2) or not np.all(np.isfinite(prev)):
        return None
    best_perm, best_cost = None, np.inf
    for perm in permutations(range(4)):
        cost = float(np.sum((points[list(perm)] - prev) ** 2))
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
    matched = None
    if previous_points is not None:
        matched = _match_previous(points, previous_points)
    points = _order_clockwise(points) if matched is None else matched
    return points, True
