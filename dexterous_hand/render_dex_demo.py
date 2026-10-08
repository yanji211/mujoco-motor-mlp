# -*- coding: utf-8 -*-
"""无头渲染灵巧手运动 GIF：open -> 强力抓握 -> 捏合 -> 张开。

优先用训练好的 MLP 学生控制器；若 model_dex.npz 不存在则退化为 PID 老师，
保证一定能渲染出"手在动"的证明。运行：py -3 render_dex_demo.py
"""
import os
import mujoco
import numpy as np
from PIL import Image

from dex_common import NJ, DT, GEAR, KP, KD, KI, GRASP_POSES

RENDER_W, RENDER_H = 720, 540
SEQ = ["张开", "强力抓握", "捏合", "张开"]
STEPS_PER_POSE = 130
FPS = 25


def make_pid():
    integ = np.zeros(NJ)
    def ctrl(th, om, tgt):
        nonlocal integ
        e = tgt - th
        integ = np.clip(integ + e * DT, -5, 5)
        return np.clip((KP * e + KI * integ - KD * om) / GEAR, -1, 1)
    def reset():
        nonlocal integ
        integ[:] = 0
    return ctrl, reset


def make_mlp():
    from hybrid_dex import HybridDex
    hy = HybridDex()  # MLP 前馈 + 小增益 PID 残差（部署形态）
    def ctrl(th, om, tgt):
        return hy.control(th, om, tgt)
    def reset():
        hy.reset()
    return ctrl, reset


def main():
    m = mujoco.MjModel.from_xml_path("dexterous_hand.xml")
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, RENDER_H, RENDER_W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.0, 0.1]
    cam.distance = 1.9
    cam.elevation = -12
    cam.azimuth = 115

    use_mlp = os.path.exists("model_dex.npz")
    ctrl, reset = (make_mlp() if use_mlp else make_pid())
    print(f"控制器: {'MLP+PID 混合(学生/部署形态)' if use_mlp else 'PID 老师(未训练)'}")
    tag = "hybrid" if use_mlp else "pid"

    mujoco.mj_resetData(m, d)
    d.qpos[:NJ] = 0.0
    reset()

    frames = []
    for pose in SEQ:
        tgt = GRASP_POSES[pose]
        print(f"  -> {pose}")
        for _ in range(STEPS_PER_POSE):
            th = d.qpos[:NJ].copy()
            om = d.qvel[:NJ].copy()
            d.ctrl[:NJ] = ctrl(th, om, tgt)
            mujoco.mj_step(m, d)
            renderer.update_scene(d, camera=cam)
            img = Image.fromarray(renderer.render())
            frames.append(img)

    out = f"dex_demo_{tag}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0)
    print(f"已保存 {out}（{len(frames)} 帧, {len(frames)/FPS:.1f}s）")


if __name__ == "__main__":
    main()
