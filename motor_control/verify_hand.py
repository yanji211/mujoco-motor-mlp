# -*- coding: utf-8 -*-
"""五指手闭环验证：MLP 学生 vs PID 老师，在多个姿态目标上对比稳态误差。"""
import mujoco
import numpy as np
from model import MLP
from generate_data_hand import KP, KI, KD, GEAR, NJ, DT

m = mujoco.MjModel.from_xml_path("hand.xml")
mlp = MLP(sizes=(21, 128, 128, 7))
mlp.load("model_hand.npz")


class PID:
    def __init__(self):
        self.integral = np.zeros(NJ)
    def reset(self):
        self.integral[:] = 0
    def control(self, th, om, tgt):
        e = tgt - th
        self.integral = np.clip(self.integral + e * DT, -5, 5)
        return np.clip((KP * e + KI * self.integral - KD * om) / GEAR, -1, 1)


def run(ctrl_fn, tgt, steps=400):
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


poses = {
    "张开": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "抓握": np.array([0.6, -0.8, -1.2, -1.2, -1.2, -1.2, 1.2]),
    "高举": np.array([1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "侧摆": np.array([-1.2, -0.5, -0.8, -0.6, -0.9, -0.7, 0.8]),
}

pid = PID()
mlp_c = type("M", (), {"reset": lambda s: None,
    "control": lambda s, th, om, tgt: mlp.forward(
        np.concatenate([th, om, tgt]).astype(np.float32).reshape(1, -1))[0][0]})()

names = "肩/肘/食/中/名/小/拇"
print(f"关节顺序: {names}")
print(f"{'姿态':<6}{'PID最大误差':<14}{'MLP最大误差':<14}{'PID逐关节':<34}{'MLP逐关节'}")
for name, tgt in poses.items():
    ep = run(pid, tgt)
    em = run(mlp_c, tgt)
    print(f"{name:<6}{ep.max():<14.4f}{em.max():<14.4f}{str(ep.round(3)):<34}{em.round(3)}")
