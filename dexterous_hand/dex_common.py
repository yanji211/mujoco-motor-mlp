# -*- coding: utf-8 -*-
"""灵巧手 14 自由度：共享参数、PID 老师增益、协调手势目标集。

关节顺序（与 dexterous_hand.xml 严格一致）：
 0 shoulder  1 elbow  2 wrist_flex  3 wrist_dev
 4 index_mcp 5 index_pip 6 middle_mcp 7 middle_pip
 8 ring_mcp  9 ring_pip 10 pinky_mcp 11 pinky_pip
12 thumb_abd 13 thumb_flex

状态 X = [θ×14, ω×14, tgt×14] (42 维)
标签 Y = [ctrl×14]            (14 维, 归一化 [-1,1])
"""
import numpy as np

NJ = 14
DT = 0.01

JOINT_NAMES = [
    "肩", "肘", "腕屈", "腕偏",
    "食MCP", "食PIP", "中MCP", "中PIP",
    "名MCP", "名PIP", "小MCP", "小PIP",
    "拇外展", "拇屈",
]
COLORS = [
    "#4aa8ff", "#ff6b6b", "#3fb950", "#d2a8ff",
    "#f0883e", "#ffd33d", "#56d364", "#e3a857",
    "#db61a2", "#a371f7", "#1f9ee3", "#7ee787",
    "#ff7b72", "#ffa657",
]

# 执行器齿轮比（实际力矩 = ctrl × gear）
GEAR = np.array([3.0, 1.5, 1.0, 1.0, 0.4, 0.35, 0.45, 0.4,
                 0.4, 0.35, 0.35, 0.3, 0.5, 0.45])

# PID 老师增益：臂/腕惯量大用中高 kp + 较强阻尼防振荡饱和；手指惯量极小、低 kp 防饱和
KP = np.array([12.0, 3.0, 2.0, 2.0, 0.25, 0.22, 0.28, 0.25,
               0.25, 0.22, 0.22, 0.18, 0.4, 0.35])
KD = np.array([2.5, 0.5, 0.25, 0.25, 0.03, 0.025, 0.03, 0.025,
               0.03, 0.025, 0.025, 0.02, 0.05, 0.04])
KI = np.array([0.6, 0.15, 0.08, 0.08, 0.015, 0.015, 0.015, 0.015,
               0.015, 0.015, 0.012, 0.012, 0.025, 0.02])


def _arr(*vals):
    return np.array(vals, dtype=np.float64)


# 协调手势目标姿态（直接对应 14 关节顺序）
GRASP_POSES = {
    "张开": _arr(0.30, -0.50, 0.30, 0.00,
                 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                 0.90, 0.00),
    "强力抓握": _arr(0.30, -0.50, 0.50, 0.00,
                     1.40, 1.80, 1.40, 1.90, 1.30, 1.80, 1.20, 1.70,
                     0.20, 1.40),
    "捏合": _arr(0.30, -0.40, 0.40, 0.00,
                 0.90, 1.10, 1.30, 1.60, 1.30, 1.60, 1.20, 1.50,
                 0.40, 1.20),
    "指向": _arr(0.30, -0.40, 0.40, 0.00,
                 0.00, 0.00, 1.40, 1.80, 1.40, 1.80, 1.30, 1.70,
                 0.60, 0.30),
}

POSE_WEIGHTS = {"强力抓握": 0.40, "张开": 0.25, "捏合": 0.15, "指向": 0.10}
# 其余 10% 为完全随机目标


def sample_target(rng, tgt_lo, tgt_hi):
    """按手势分布采样一个 14 维目标，含轻微抖动，保证落在限位内。"""
    r = rng.random()
    cum = 0.0
    for name, w in POSE_WEIGHTS.items():
        cum += w
        if r < cum:
            base = GRASP_POSES[name].copy()
            jitter = rng.uniform(-0.12, 0.12, NJ)
            # 只对四指/拇指的弯曲关节加抖动，腕/臂保持稳定
            base = np.clip(base + jitter, tgt_lo, tgt_hi)
            return base
    # 完全随机
    return rng.uniform(tgt_lo, tgt_hi)
