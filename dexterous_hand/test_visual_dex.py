# -*- coding: utf-8 -*-
"""灵巧手中文仪表盘：MLP 学生 vs PID 老师。

- 14 关节：肩/肘/腕屈/腕偏 + 食中名小四指(MCP,PIP) + 拇外展/拇屈
- 左屏：MuJoCo 实时渲染；右屏：14 关节角度跟踪曲线 + 目标虚线
- 按钮：启动/暂停、复位、切换控制源、四个协调手势（张开/强力抓握/捏合/指向）
运行：py -3 test_visual_dex.py  （需本机有显示器）
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
from hybrid_dex import HybridDex
from dex_common import NJ, DT, GEAR, KP, KD, KI, GRASP_POSES, JOINT_NAMES, COLORS

FPS_MS = 16
RENDER_W, RENDER_H = 560, 460
PLOT_LEN = 400


class PIDSolver:
    def __init__(self):
        self.integral = np.zeros(NJ)
    def reset(self):
        self.integral[:] = 0
    def control(self, th, om, tgt):
        e = tgt - th
        self.integral = np.clip(self.integral + e * DT, -5, 5)
        return np.clip((KP * e + KI * self.integral - KD * om) / GEAR, -1, 1)


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("灵巧手 14 自由度 —— PID 老师 vs MLP 学生")
        root.resizable(False, False)

        self.model = mujoco.MjModel.from_xml_path("dexterous_hand.xml")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.55, 0.0, 0.1]
        self.camera.distance = 1.9
        self.camera.elevation = -12
        self.camera.azimuth = 115

        self.mlp = MLP(sizes=(42, 256, 256, 14))
        try:
            self.mlp.load("model_dex.npz")
            self.has_mlp = True
            self.hy = HybridDex()
        except FileNotFoundError:
            self.has_mlp = False
            self.hy = None
        self.pid = PIDSolver()
        self.mode = 0  # 0=PID老师 1=纯MLP 2=混合
        self.tgt = GRASP_POSES["张开"].copy()
        self.hist = {k: np.full(PLOT_LEN, np.nan) for k in range(NJ)}
        self.tgt_hist = {k: np.full(PLOT_LEN, np.nan) for k in range(NJ)}
        self.step_count = 0
        self.running = False

        self._build_ui()
        self._reset()

    def _build_ui(self):
        tk.Label(self.root, anchor="w", justify="left", wraplength=1080,
                 text="灵巧手 14 自由度：肩+肘+双自由度腕+掌，四指各 MCP/PIP 两节可弯，拇指可外展对捏。\n"
                      "切换控制源看 MLP 学生能否复刻 PID 老师完成协调手势——模仿发生在训练阶段。",
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

        self.fig = Figure(figsize=(5.6, 3.4), dpi=100, facecolor="#161b22")
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.figc = FigureCanvasTkAgg(self.fig, master=self.root)
        self.figc.get_tk_widget().grid(row=2, column=0, padx=8, pady=(0, 8))

        panel = ttk.LabelFrame(self.root, text="控制面板", padding=12)
        panel.grid(row=1, column=1, sticky="n", padx=(0, 8), pady=8)
        self.btn_start = ttk.Button(panel, text="启动", command=self._toggle, width=14)
        self.btn_start.grid(row=0, column=0, sticky="ew", pady=4)
        ttk.Button(panel, text="复位", command=self._reset, width=14).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(panel, text="切换控制源（老师/学生/混合）", command=self._switch_src).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(panel, text="目标手势：").grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 2))
        r = 3
        for name in ["张开", "强力抓握", "捏合", "指向"]:
            ttk.Button(panel, text=name, width=14,
                       command=lambda n=name: self._set_pose(n)).grid(
                row=r, column=0, columnspan=2, sticky="ew", pady=2)
            r += 1

        status = ttk.LabelFrame(self.root, text="各关节角度 θ / 目标", padding=10)
        status.grid(row=2, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.vars = {}
        for j in range(NJ):
            ttk.Label(status, text=JOINT_NAMES[j], foreground=COLORS[j]).grid(
                row=j, column=0, sticky="w", padx=(0, 8))
            v = tk.StringVar(value="—")
            self.vars[j] = v
            ttk.Label(status, textvariable=v, foreground="#8b949e").grid(row=j, column=1, sticky="e")

        self.hint = tk.StringVar(value="提示：点「启动」后切到 MLP 学生，看它能否独立完成抓取手势")
        ttk.Label(self.root, textvariable=self.hint).grid(row=3, column=1, sticky="w", padx=(0, 8))

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#161b22")
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.set_title("14 关节角度跟踪（虚线=目标）", color="#e6edf3", fontsize=9)
        ax.grid(True, color="#30363d", lw=0.4)

    def _on_drag(self, event):
        if self._last_mouse is None:
            return
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        self.camera.azimuth += dx * 0.5
        self.camera.elevation = max(-89, min(89, self.camera.elevation - dy * 0.5))
        self._last_mouse = (event.x, event.y)

    def _on_wheel(self, event):
        self.camera.distance = max(0.5, min(6.0, self.camera.distance * (0.9 if event.delta > 0 else 1.1)))

    def _toggle(self):
        self.running = not self.running
        self.btn_start.config(text="暂停" if self.running else "启动")

    def _switch_src(self):
        if not self.has_mlp:
            self.hint.set("尚未训练 model_dex.npz，无法切换学生")
            return
        self.mode = (self.mode + 1) % 3
        self.pid.reset()
        if self.hy is not None:
            self.hy.reset()
        if self.mode == 0:
            self.src_label.config(text="控制源：PID 老师", fg="#4aa8ff")
            self.hint.set("PID 老师（基准）")
        elif self.mode == 1:
            self.src_label.config(text="控制源：纯 MLP 学生", fg="#ff6b6b")
            self.hint.set("纯 MLP 前馈——注意近零区域可能有微小稳态偏移")
        else:
            self.src_label.config(text="控制源：MLP+PID 混合", fg="#56d364")
            self.hint.set("混合：MLP 前馈 + PID 残差清零稳态误差（部署形态）")

    def _set_pose(self, name):
        self.tgt = GRASP_POSES[name].copy()
        self.hint.set(f"目标姿态：{name}")

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
        if self.mode == 1 and self.has_mlp:
            x = np.concatenate([th, om, self.tgt]).astype(np.float32).reshape(1, -1)
            u = self.mlp.forward(x)[0][0]
        elif self.mode == 2 and self.has_mlp:
            u = self.hy.control(th, om, self.tgt)
        else:
            u = self.pid.control(th, om, self.tgt)
        self.data.ctrl[:NJ] = np.clip(u, -1, 1)
        if self.running:
            mujoco.mj_step(self.model, self.data)
        return th

    def _update(self, th):
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
            ax.plot(x, self.hist[j], color=COLORS[j], lw=0.9, label=JOINT_NAMES[j])
            ax.plot(x, self.tgt_hist[j], color=COLORS[j], lw=0.5, ls="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=6, facecolor="#161b22",
                  labelcolor="#e6edf3", edgecolor="#30363d", ncol=3)
        self.figc.draw_idle()

    def _loop(self):
        th = self._step_once()
        self._update(th)
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
