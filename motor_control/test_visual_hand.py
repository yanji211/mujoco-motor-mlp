# -*- coding: utf-8 -*-
"""多关节手臂 MLP vs PID 对比仪表盘。

- 4 关节：肩(shoulder) / 肘(elbow) / 指(finger) / 拇(thumb)
- 左半屏：PID 老师手臂；右半屏：MLP 学生手臂（同一 hand.xml 渲染两次不同 qpos）
  简化：单臂场景，可切换 老师/学生/混合 三种控制源，避免维护双份模型。
- 实时 4 关节角度曲线 + 目标虚线
运行：py -3 test_visual_hand.py
"""
import tkinter as tk
from tkinter import ttk

import mujoco
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from model import MLP
from generate_data_hand import KP, KI, KD, GEAR, NJ, DT

FPS_MS = 16
RENDER_W, RENDER_H = 560, 400
PLOT_LEN = 400
JOINT_NAMES = ["肩", "肘", "食", "中", "名", "小", "拇"]
COLORS = ["#4aa8ff", "#ff6b6b", "#3fb950", "#d2a8ff", "#f0883e", "#db61a2", "#56d364"]


class PIDSolver:
    """多关节解耦 PID（与数据生成完全一致），用于"老师"模式。"""
    def __init__(self):
        self.integral = np.zeros(NJ)

    def reset(self):
        self.integral[:] = 0

    def control(self, th, om, tgt):
        e = tgt - th
        self.integral = np.clip(self.integral + e * DT, -5, 5)
        tau = KP * e + KI * self.integral - KD * om
        return np.clip(tau / GEAR, -1, 1)


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("多关节手臂 —— PID 老师 vs MLP 学生")
        root.resizable(False, False)

        self.model = mujoco.MjModel.from_xml_path("hand.xml")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.3, 0.0, 0.1]
        self.camera.distance = 1.6
        self.camera.elevation = -25
        self.camera.azimuth = 120

        self.mlp = MLP(sizes=(21, 128, 128, 7))
        self.mlp.load("model_hand.npz")
        self.pid = PIDSolver()

        # 控制源：0=PID 老师  1=MLP 学生
        self.use_mlp = False
        # 目标姿态：默认"抓握"（指闭合、拇合拢、肩肘伸向方块）
        self.targets = tk.DoubleVar(value=0)
        self.tgt = np.array([0.6, -0.8, -1.2, -1.2, -1.2, -1.2, 1.2])   # 抓握方块姿态
        self.open_tgt = np.zeros(7)  # 张开姿态
        self.hist = {k: np.full(PLOT_LEN, np.nan) for k in range(NJ)}
        self.tgt_hist = {k: np.full(PLOT_LEN, np.nan) for k in range(NJ)}
        self.step_count = 0
        self.running = False

        self._build_ui()
        self._reset()

    def _build_ui(self):
        tk.Label(self.root, anchor="w", justify="left", wraplength=1040,
                 text="五指手 7 关节：肩(摆动) + 肘(弯曲) + 食/中/名/小四指 + 拇(对捏)。切换控制源看 MLP 学生能否复刻 PID 老师抓握。\n"
                      "模仿发生在训练阶段——网络学的是「看到这套关节状态、要去这个目标，七个关节各出多大力」。",
                 bg="#1b2028", fg="#ffd580", font=("", 9)).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))

        self.canvas = tk.Canvas(self.root, width=RENDER_W, height=RENDER_H, bg="#10151c")
        self.canvas.grid(row=1, column=0, padx=8, pady=8)
        self._last_mouse = None
        self.canvas.bind("<ButtonPress-1>", lambda e: setattr(self, "_last_mouse", (e.x, e.y)))
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.src_label = tk.Label(self.root, text="控制源：PID 老师", bg="#10151c", fg="#4aa8ff")
        self.src_label.grid(row=1, column=0, sticky="s", pady=(0, 14))

        self.fig = Figure(figsize=(5.6, 2.8), dpi=100, facecolor="#161b22")
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.figc = FigureCanvasTkAgg(self.fig, master=self.root)
        self.figc.get_tk_widget().grid(row=2, column=0, padx=8, pady=(0, 8))

        panel = ttk.LabelFrame(self.root, text="控制面板", padding=12)
        panel.grid(row=1, column=1, sticky="n", padx=(0, 8), pady=8)

        self.btn_start = ttk.Button(panel, text="启动", command=self._toggle, width=12)
        self.btn_start.grid(row=0, column=0, sticky="ew", pady=4)
        ttk.Button(panel, text="复位", command=self._reset, width=12).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(panel, text="切换控制源（老师/学生）", command=self._switch_src).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(panel, text="抓握方块", command=lambda: self._set_pose("grasp")).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(panel, text="张开手臂", command=lambda: self._set_pose("open")).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=4)

        status = ttk.LabelFrame(self.root, text="各关节角度 θ / 目标", padding=10)
        status.grid(row=2, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.vars = {}
        for j in range(NJ):
            ttk.Label(status, text=JOINT_NAMES[j], foreground=COLORS[j]).grid(row=j, column=0, sticky="w", padx=(0, 8))
            v = tk.StringVar(value="—")
            self.vars[j] = v
            ttk.Label(status, textvariable=v, foreground="#8b949e").grid(row=j, column=1, sticky="e")

        self.hint = tk.StringVar(value="提示：点「启动」后切到 MLP 学生，看它能否独立抓握")
        ttk.Label(self.root, textvariable=self.hint).grid(row=3, column=1, sticky="w", padx=(0, 8))

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#161b22")
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.set_title("4 关节角度跟踪（虚线=目标）", color="#e6edf3", fontsize=9)
        ax.grid(True, color="#30363d", lw=0.4)

    def _on_drag(self, event):
        if self._last_mouse is None:
            return
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        self.camera.azimuth += dx * 0.5
        self.camera.elevation = max(-89, min(89, self.camera.elevation - dy * 0.5))
        self._last_mouse = (event.x, event.y)

    def _on_wheel(self, event):
        self.camera.distance = max(0.4, min(5.0, self.camera.distance * (0.9 if event.delta > 0 else 1.1)))

    def _toggle(self):
        self.running = not self.running
        self.btn_start.config(text="暂停" if self.running else "启动")

    def _switch_src(self):
        self.use_mlp = not self.use_mlp
        self.pid.reset()
        if self.use_mlp:
            self.src_label.config(text="控制源：MLP 学生", fg="#ff6b6b")
            self.hint.set("现在是 MLP 学生在控制整条手臂——注意它没被针对当前目标专门调过")
        else:
            self.src_label.config(text="控制源：PID 老师", fg="#4aa8ff")
            self.hint.set("切回 PID 老师")

    def _set_pose(self, which):
        self.tgt = self.tgt if which == "grasp" else self.open_tgt
        self.hint.set(f"目标姿态切换为：{'抓握方块' if which == 'grasp' else '完全张开'}")

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:NJ] = 0.0
        self.data.qvel[:NJ] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.pid.reset()
        for k in self.hist:
            self.hist[k][:] = np.nan
            self.tgt_hist[k][:] = np.nan
        self.step_count = 0

    def _step_once(self):
        th = self.data.qpos[:NJ].copy()
        om = self.data.qvel[:NJ].copy()
        if self.use_mlp:
            x = np.concatenate([th, om, self.tgt]).astype(np.float32).reshape(1, -1)
            u = self.mlp.forward(x)[0][0]
        else:
            u = self.pid.control(th, om, self.tgt)
        self.data.ctrl[:NJ] = np.clip(u, -1, 1)
        if self.running:
            mujoco.mj_step(self.model, self.data)
        return th, om, u

    def _update(self, th, u):
        i = self.step_count % PLOT_LEN
        for j in range(NJ):
            self.hist[j][i] = th[j]
            self.tgt_hist[j][i] = self.tgt[j]
            self.vars[j].set(f"{th[j]:+.3f} / {self.tgt[j]:+.2f}")
        self.step_count += 1

    def _redraw(self):
        ax = self.ax
        ax.clear()
        self._style_ax()
        x = np.arange(PLOT_LEN)
        for j in range(NJ):
            ax.plot(x, self.hist[j], color=COLORS[j], lw=1.0, label=JOINT_NAMES[j])
            ax.plot(x, self.tgt_hist[j], color=COLORS[j], lw=0.6, ls="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=7, facecolor="#161b22",
                  labelcolor="#e6edf3", edgecolor="#30363d", ncol=2)
        self.figc.draw_idle()

    def _loop(self):
        th, om, u = self._step_once()
        self._update(th, u)
        self.renderer.update_scene(self.data, camera=self.camera)
        img = Image.fromarray(self.renderer.render())
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        if self.step_count % 3 == 0:
            self._redraw()
        self.root.after(FPS_MS, self._loop)

    def run(self):
        self._loop()
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    Dashboard(root).run()
