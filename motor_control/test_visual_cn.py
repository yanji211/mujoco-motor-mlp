"""PID vs MLP 对比仪表盘：双臂并排 + 实时曲线 + 扰动测试。

- 蓝臂：PID 老师控制
- 红臂：训练出的 MLP 控制（model_motor.npz）
- 两条臂动力学完全相同，直观检验 MLP 是否真的学会了控制
- "施加扰动"按钮给两条臂相同的角度+速度冲击，看谁拉得回来
运行：py -3 test_visual_cn.py
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
from pid import PID

FPS_MS = 16          # 约 60Hz 刷新
RENDER_W, RENDER_H = 520, 380
PLOT_LEN = 600       # 曲线保留最近 600 步（6 秒）


class Dashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("电机控制对比仪表盘 —— PID 老师 vs MLP 学生")
        root.resizable(False, False)

        # 仿真：双臂模型
        self.model = mujoco.MjModel.from_xml_path("motor_dual.xml")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.25, 0.35, 0.1]
        self.camera.distance = 2.4
        self.camera.elevation = -40
        self.camera.azimuth = 70

        # 控制器
        self.mlp = MLP(sizes=(3, 32, 32, 1))
        self.mlp.load("model_motor.npz")
        # 加大积分增益：演示 PID 靠积分抵消重力/负载稳态误差
        self.pid = PID(ki=2.0)

        self.running = False
        self.gravity_on = False   # 重力开关（下拉力矩）
        self.load_on = False       # 额外负载开关
        self.GRAVITY_TORQUE = -0.3  # N·m，模拟重力对关节的下拉
        self.LOAD_TORQUE = -0.3     # N·m，额外挂负载
        self.target = tk.DoubleVar(value=1.0)
        self.hist = {k: np.full(PLOT_LEN, np.nan) for k in
                     ("th_pid", "th_mlp", "tg")}
        self.step_count = 0

        self._build_ui()
        self._reset()

    # ---------- 界面 ----------
    def _build_ui(self):
        # 顶部一句话说明
        tk.Label(self.root, anchor="w", justify="left", wraplength=980,
                 text="红臂听“神经网络”指挥，蓝臂听“PID 公式”指挥——两臂互相独立、没有对讲机。\n"
                      "它们动作一致，说明训练出的网络成功学会了老师的控制手艺（模仿发生在训练阶段，不是运行时跟随）。",
                 bg="#1b2028", fg="#ffd580", font=("", 9)).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))

        # 左上：渲染画面
        self.canvas = tk.Canvas(self.root, width=RENDER_W, height=RENDER_H, bg="#10151c")
        self.canvas.grid(row=1, column=0, padx=8, pady=8)
        tk.Label(self.root, text="蓝臂 = PID 老师    红臂 = MLP 学生",
                 bg="#10151c", fg="#e6edf3").grid(row=1, column=0, sticky="s", pady=(0, 14))

        # 左下：实时曲线
        self.fig = Figure(figsize=(5.2, 2.6), dpi=100, facecolor="#161b22")
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.figc = FigureCanvasTkAgg(self.fig, master=self.root)
        self.figc.get_tk_widget().grid(row=2, column=0, padx=8, pady=(0, 8))

        # 右侧：控制面板
        panel = ttk.LabelFrame(self.root, text="控制面板", padding=12)
        panel.grid(row=1, column=1, sticky="n", padx=(0, 8), pady=8)

        ttk.Label(panel, text="目标角度 (弧度):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(panel, from_=-3.14, to=3.14, increment=0.1,
                    textvariable=self.target, width=9).grid(row=0, column=1, sticky="w", padx=(6, 0))
        quick = ttk.Combobox(panel, values=["-1.5", "-0.5", "0.5", "1.0", "1.5"], width=9)
        quick.set("1.0")
        quick.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(panel, text="常用角度:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        quick.bind("<<ComboboxSelected>>", lambda e: self.target.set(float(quick.get())))

        btns = ttk.Frame(panel)
        btns.grid(row=2, column=0, columnspan=2, pady=14)
        self.btn_start = ttk.Button(btns, text="启动", command=self._toggle, width=9)
        self.btn_start.grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="复位", command=self._reset, width=9).grid(row=0, column=1, padx=4)
        ttk.Button(panel, text="施加扰动（两臂同时踹飞）",
                   command=self._kick).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(panel, text="只显示学生臂（藏起老师）",
                   command=self._toggle_pid_view).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        # 重力 / 负载开关
        self.btn_gravity = ttk.Button(panel, text="重力：关", command=self._toggle_gravity)
        self.btn_gravity.grid(row=5, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))
        self.btn_load = ttk.Button(panel, text="负载：关", command=self._toggle_load)
        self.btn_load.grid(row=5, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))

        # 状态表
        status = ttk.LabelFrame(self.root, text="实时状态", padding=10)
        status.grid(row=2, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.vars = {}
        rows = [("θ (rad)", "th"), ("θ 误差", "err"), ("ω (rad/s)", "om"), ("力矩 (N·m)", "u")]
        ttk.Label(status, text="PID", foreground="#4aa8ff").grid(row=0, column=1, padx=10)
        ttk.Label(status, text="MLP", foreground="#ff6b6b").grid(row=0, column=2, padx=10)
        for r, (name, key) in enumerate(rows, start=1):
            ttk.Label(status, text=name).grid(row=r, column=0, sticky="w")
            for c, ctrl in enumerate(("pid", "mlp"), start=1):
                v = tk.StringVar(value="—")
                self.vars[f"{key}_{ctrl}"] = v
                ttk.Label(status, textvariable=v, foreground="#8b949e").grid(row=r, column=c, sticky="e")
        ttk.Label(status, text="两臂角度差 Δθ:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.dtheta_var = tk.StringVar(value="—")
        ttk.Label(status, textvariable=self.dtheta_var, foreground="#3fb950",
                  font=("", 10, "bold")).grid(row=5, column=1, columnspan=2, sticky="e", pady=(6, 0))

        self.hint = tk.StringVar(value="提示：先点“启动”，再试试“施加扰动”")
        ttk.Label(self.root, textvariable=self.hint).grid(row=3, column=1, sticky="w", padx=(0, 8))

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#161b22")
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.set_title("角度跟踪（虚线=目标）", color="#e6edf3", fontsize=9)
        ax.grid(True, color="#30363d", lw=0.4)

    # ---------- 仿真逻辑 ----------
    def _toggle(self):
        self.running = not self.running
        self.btn_start.config(text="暂停" if self.running else "启动")

    def _toggle_gravity(self):
        self.gravity_on = not self.gravity_on
        self.btn_gravity.config(text=f"重力：{'开' if self.gravity_on else '关'}")
        self.hint.set(f"重力力矩已{'施加' if self.gravity_on else '移除'}（-0.3 N·m 持续下拉）——"
                      "观察：PID 靠积分消除误差，MLP 留有稳态误差")

    def _toggle_load(self):
        self.load_on = not self.load_on
        self.btn_load.config(text=f"负载：{'开' if self.load_on else '关'}")
        self.hint.set(f"额外负载已{'挂上' if self.load_on else '取下'}（再 -0.3 N·m）")

    def _toggle_pid_view(self):
        """隐藏/显示蓝臂（PID 老师），验证红臂不依赖蓝臂也能独立工作。
        geom 索引：0=底座 1=蓝臂 2=红臂。"""
        self.pid_hidden = not getattr(self, "pid_hidden", False)
        self.model.geom_rgba[1, 3] = 0.0 if self.pid_hidden else 1.0
        if self.pid_hidden:
            self.hint.set("蓝臂已隐藏——红臂仍在独立工作，说明它不是跟着蓝臂动，而是自己会控制")
        else:
            self.hint.set("蓝臂已显示，恢复双臂对比")

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.pid.reset()
        for k in self.hist:
            self.hist[k][:] = np.nan
        self.step_count = 0

    def _kick(self):
        """给两条臂施加完全相同的冲击：位置偏移 + 反向初速度。"""
        tg = self.target.get()
        kick = 0.8 * (1.0 if tg >= 0 else -1.0)
        self.data.qpos[0] += kick
        self.data.qpos[1] += kick
        self.data.qvel[0] -= 2.0 * np.sign(kick)
        self.data.qvel[1] -= 2.0 * np.sign(kick)
        if not self.running:
            self._toggle()
        self.hint.set(f"已施加扰动：+{kick:.1f} rad 位移、-2 rad/s 初速，观察两臂恢复")

    def _step_once(self):
        tg = self.target.get()
        # 重力/负载：以常值扰动力矩施加到两个关节（模拟重力下拉）
        disturbance = (self.GRAVITY_TORQUE if self.gravity_on else 0.0) + \
                      (self.LOAD_TORQUE if self.load_on else 0.0)
        self.data.qfrc_applied[0] = disturbance
        self.data.qfrc_applied[1] = disturbance

        th_p, om_p = self.data.qpos[0], self.data.qvel[0]
        th_m, om_m = self.data.qpos[1], self.data.qvel[1]

        u_pid = self.pid.control(th_p, om_p, tg)
        u_mlp = float(self.mlp.forward(
            np.array([[th_m, om_m, tg]], dtype=np.float32))[0][0, 0])

        self.data.ctrl[0] = np.clip(u_pid, -1, 1)
        self.data.ctrl[1] = np.clip(u_mlp, -1, 1)
        if self.running:
            mujoco.mj_step(self.model, self.data)
        return (th_p, om_p, u_pid), (th_m, om_m, u_mlp), tg

    def _update_history(self, pid_s, mlp_s, tg):
        i = self.step_count % PLOT_LEN
        self.hist["th_pid"][i] = pid_s[0]
        self.hist["th_mlp"][i] = mlp_s[0]
        self.hist["tg"][i] = tg
        self.step_count += 1

    def _redraw_plot(self):
        ax = self.ax
        ax.clear()
        self._style_ax()
        x = np.arange(PLOT_LEN)
        ax.plot(x, self.hist["th_pid"], color="#4aa8ff", lw=1.0, label="PID")
        ax.plot(x, self.hist["th_mlp"], color="#ff6b6b", lw=1.0, label="MLP")
        ax.plot(x, self.hist["tg"], color="#8b949e", lw=0.8, ls="--", label="目标")
        ax.legend(loc="lower right", fontsize=7, facecolor="#161b22",
                  labelcolor="#e6edf3", edgecolor="#30363d")
        self.figc.draw_idle()

    def _update_status(self, pid_s, mlp_s):
        for (th, om, u), ctrl in ((pid_s, "pid"), (mlp_s, "mlp")):
            self.vars[f"th_{ctrl}"].set(f"{th:+.3f}")
            self.vars[f"err_{ctrl}"].set(f"{abs(th - self.target.get()):.4f}")
            self.vars[f"om_{ctrl}"].set(f"{om:+.3f}")
            self.vars[f"u_{ctrl}"].set(f"{u:+.3f}")
        d = abs(pid_s[0] - mlp_s[0])
        self.dtheta_var.set(f"{d:.4f} rad")

    def _loop(self):
        pid_s, mlp_s, tg = self._step_once()
        self._update_history(pid_s, mlp_s, tg)
        self._update_status(pid_s, mlp_s)

        self.renderer.update_scene(self.data, camera=self.camera)
        img = Image.fromarray(self.renderer.render())
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        if self.step_count % 3 == 0:
            self._redraw_plot()

        self.root.after(FPS_MS, self._loop)

    def run(self):
        self._loop()
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    Dashboard(root).run()
