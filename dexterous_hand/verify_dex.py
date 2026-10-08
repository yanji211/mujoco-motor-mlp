# -*- coding: utf-8 -*-
"""灵巧手闭环验证：纯 MLP 学生 vs PID 老师 vs MLP+PID 混合，对比稳态误差。"""
import mujoco
import numpy as np
from model import MLP
from hybrid_dex import HybridDex
from dex_common import NJ, DT, GEAR, KP, KD, KI, GRASP_POSES, JOINT_NAMES


m = mujoco.MjModel.from_xml_path("dexterous_hand.xml")
mlp = MLP(sizes=(42, 256, 256, 14)); mlp.load("model_dex.npz")


class PID:
    def __init__(self):
        self.integral = np.zeros(NJ)
    def reset(self):
        self.integral[:] = 0
    def control(self, th, om, tgt):
        e = tgt - th
        self.integral = np.clip(self.integral + e * DT, -5, 5)
        return np.clip((KP * e + KI * self.integral - KD * om) / GEAR, -1, 1)


def make_mlp_ctrl():
    class C:
        def reset(self):
            pass
        def control(self, th, om, tgt):
            x = np.concatenate([th, om, tgt]).astype(np.float32).reshape(1, -1)
            return mlp.forward(x)[0][0]
    return C()


def run(ctrl_fn, tgt, steps=500):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:NJ] = 0.0
    d.qvel[:NJ] = 0.0
    ctrl_fn.reset()
    for _ in range(steps):
        th, om = d.qpos[:NJ].copy(), d.qvel[:NJ].copy()
        d.ctrl[:NJ] = ctrl_fn.control(th, om, tgt)
        mujoco.mj_step(m, d)
    return np.abs(d.qpos[:NJ] - tgt)


pid = PID()
mlp_c = make_mlp_ctrl()
hy = HybridDex()

print(f"关节顺序: {'/'.join(JOINT_NAMES)}")
print(f"{'姿态':<10}{'PID最大':<10}{'MLP最大':<10}{'混合最大':<10}{'混合RMS'}")
tp = tm = th_ = 0.0
for name, tgt in GRASP_POSES.items():
    ep = run(pid, tgt)
    em = run(mlp_c, tgt)
    eh = run(hy, tgt)
    tp += ep.max(); tm += em.max(); th_ += eh.max()
    print(f"{name:<10}{ep.max():<10.4f}{em.max():<10.4f}{eh.max():<10.4f}{np.sqrt((eh**2).mean()):.4f}")
print(f"\n平均最大误差: PID={tp/4:.4f}  MLP={tm/4:.4f}  混合={th_/4:.4f}")
