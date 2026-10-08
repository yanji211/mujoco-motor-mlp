# -*- coding: utf-8 -*-
"""灵巧手数据生成：14 路 PID 老师并行控制，采 (42维状态 -> 14维力矩)。

关节顺序见 dex_common.JOINT_NAMES。每关节独立 PID，但目标来自协调手势分布，
数据里自然包含"别的关节在动/重力耦合时我该出多大力"——网络学到耦合补偿。
"""
import mujoco
import numpy as np

from dex_common import NJ, DT, GEAR, KP, KD, KI, sample_target

N_EPISODES = 4000
STEPS = 250
MARGIN = 0.15  # 目标采样离限位的安全边距


def main():
    m = mujoco.MjModel.from_xml_path("dexterous_hand.xml")
    d = mujoco.MjData(m)
    assert m.nu == NJ and m.nq == NJ, f"关节数不符 nq={m.nq} nu={m.nu}"

    lo = m.jnt_range[:NJ, 0] + MARGIN
    hi = m.jnt_range[:NJ, 1] - MARGIN

    rng = np.random.default_rng(42)
    Xs, Ys = [], []

    for ep in range(N_EPISODES):
        mujoco.mj_resetData(m, d)
        d.qpos[:NJ] = rng.uniform(lo, hi)
        d.qvel[:NJ] = rng.uniform(-0.5, 0.5, NJ)
        tgt = sample_target(rng, lo, hi)
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
    np.savez("data/dex_data.npz", X=X, Y=Y)
    print(f"生成完成: X{X.shape} Y{Y.shape}  (~{X.nbytes/1e6:.0f}MB + {Y.nbytes/1e6:.0f}MB)")
    print(f"ctrl 饱和率(按关节 %): {(np.mean(np.abs(Y) > 0.98, axis=0) * 100).round(1)}")
    print(f"θ 范围/关节: {X[:, :NJ].min(0).round(2)} ~ {X[:, :NJ].max(0).round(2)}")


if __name__ == "__main__":
    main()
