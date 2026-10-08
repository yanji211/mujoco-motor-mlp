# -*- coding: utf-8 -*-
"""五指手数据生成：7 路 PID 老师并行控制，采 (21维状态 -> 7维力矩)。

关节顺序：shoulder / elbow / finger_index / finger_middle / finger_ring / finger_pinky / thumb
状态 X = [θ×7, ω×7, tgt×7]  (21 维)
标签 Y = [τ×7]              (7 维, 归一化 ctrl [-1,1])

多关节耦合：每关节独立 PID，数据里自然包含"别的关节在动时我该出多大力"，
网络学到耦合补偿——这正是模仿学习的价值。
"""
import mujoco
import numpy as np

N_EPISODES = 10000
STEPS = 300
DT = 0.01
NJ = 7  # 关节数

# 执行器齿轮比（与 hand.xml 的 gear 一致）：实际力矩 = ctrl × gear
GEAR = np.array([3.0, 1.5, 0.5, 0.5, 0.5, 0.5, 0.6])

# 增益：肩/肘沿用单臂整定值；四指+拇指惯量极小（强过阻尼），用低 kp 防饱和
KP = np.array([20.0, 4.5, 0.3, 0.3, 0.3, 0.3, 0.4])
KD = np.array([3.0, 0.3, 0.05, 0.05, 0.05, 0.05, 0.06])
KI = np.array([0.8, 0.2, 0.02, 0.02, 0.02, 0.02, 0.03])

# 目标角度采样范围（限位内 ±0.3 安全边距）
TGT_LO = np.array([-2.0, -1.6, -1.4, -1.4, -1.4, -1.4, 0.0])
TGT_HI = np.array([2.0, 0.8, 0.0, 0.0, 0.0, 0.0, 1.4])

# 抓握姿态：手指闭合、拇指对捏
GRASP = np.array([0.6, -0.8, -1.2, -1.2, -1.2, -1.2, 1.2])


def main():
    m = mujoco.MjModel.from_xml_path("hand.xml")
    d = mujoco.MjData(m)
    assert m.nu == NJ and m.nq == NJ, f"关节数不符 nq={m.nq} nu={m.nu}"

    rng = np.random.default_rng(42)
    Xs, Ys = [], []

    for ep in range(N_EPISODES):
        mujoco.mj_resetData(m, d)
        lo = m.jnt_range[:NJ, 0] + 0.2
        hi = m.jnt_range[:NJ, 1] - 0.2
        d.qpos[:NJ] = rng.uniform(lo, hi)
        d.qvel[:NJ] = rng.uniform(-0.5, 0.5, NJ)
        # 目标分布：40% 抓握（四指协同闭合+拇指对捏），25% 张开（全零附近），
        # 35% 随机独立目标
        r = rng.random()
        if r < 0.4:
            close = rng.uniform(0.6, 1.3)
            jitter = rng.uniform(-0.15, 0.15, 4)
            tgt = np.array([rng.uniform(-2, 2), rng.uniform(-1.5, 0.5),
                            -(close + jitter[0]), -(close + jitter[1]),
                            -(close + jitter[2]), -(close + jitter[3]),
                            rng.uniform(0.6, 1.3)])
        elif r < 0.65:
            # 张开姿态：全零附近抖动，覆盖"手指伸直"区域
            tgt = rng.uniform(-0.25, 0.25, NJ)
            tgt[0] = rng.uniform(-2, 2)
            tgt[1] = rng.uniform(-1.5, 0.5)
        else:
            tgt = rng.uniform(TGT_LO, TGT_HI)
        integral = np.zeros(NJ)

        for _ in range(STEPS):
            th, om = d.qpos[:NJ].copy(), d.qvel[:NJ].copy()
            e = tgt - th
            integral = np.clip(integral + e * DT, -5, 5)
            tau = KP * e + KI * integral - KD * om
            u = np.clip(tau / GEAR, -1, 1)   # 物理力矩 -> 归一化 ctrl
            d.ctrl[:NJ] = u
            mujoco.mj_step(m, d)
            Xs.append(np.concatenate([th, om, tgt]))
            Ys.append(u)

    X = np.array(Xs, dtype=np.float32)
    Y = np.array(Ys, dtype=np.float32)
    np.savez("data/hand_data.npz", X=X, Y=Y)
    print(f"生成完成: X{X.shape} Y{Y.shape}")
    print(f"ctrl 饱和率(按关节): {(np.mean(np.abs(Y) > 0.98, axis=0) * 100).round(1)}%")
    print(f"θ 范围: {X[:, :NJ].min(0).round(2)} ~ {X[:, :NJ].max(0).round(2)}")


if __name__ == "__main__":
    main()
