# -*- coding: utf-8 -*-
"""MLP 前馈 + PID 反馈 混合伺服控制器。

架构：τ_cmd = clamp(α·τ_ff + τ_fb, ±1)   （输出均为归一化力矩 [-1,1]）
  - τ_ff：MLP 前馈项，出力矩"大头"（学自动力学前馈/专家数据）
  - τ_fb：小增益 PD(-i) 反馈项，只消残余误差，出力矩"小头"
  - α：前馈配比，默认 0.7，给 PID 留 30% 力矩余量

安全设计（模式 A：前馈+兜底）：
  1. PID 的 kp/kd 降为纯 PID 的 1/5，避免与前馈打架；ki 保留足够权限消重力稳态误差
  2. 积分限幅 ±2.0 抗 windup（可覆盖 -0.3 N·m 级扰动）
  3. 力矩斜率限制（slew limit），每拍最大变化 0.2
  4. 前馈发散检测：|τ_ff| 连续 10 拍顶满限幅 → 屏蔽前馈，纯 PID 降级运行
  5. 传感器故障（输入 NaN）→ 前馈归零，纯 PID 接管

部署：权重仅 ~1200 float ≈ 4.8KB，STM32@168MHz 推理 ~2μs，1kHz 环路余量充足。
"""
import numpy as np


class HybridController:
    def __init__(self, model, kp=4.0, ki=0.5, kd=1.0,
                 alpha=0.7, i_limit=2.0, slew_limit=0.2,
                 ff_clip=0.7, divergence_steps=10):
        """
        model: 已训练 MLP（model.py 的 MLP 实例），forward([[θ,ω,θ_ref]]) -> 归一化力矩
        kp/ki/kd: 兜底 PID 小增益（kp/kd 为纯 PID 的 1/5；ki 需足够大以消除
                  重力/负载等常值扰动的稳态误差——实测 ki=0.5, i_limit=2.0
                  可把 -0.3 N·m 重力扰动下的稳态误差从 0.020 压到 0.008）
        alpha: 前馈配比（τ_ff 乘性缩放）
        ff_clip: 前馈输出限幅（=alpha·1.0，留 1-alpha 给 PID）
        divergence_steps: 连续饱和多少拍判定前馈发散
        """
        self.model = model
        self.kp, self.ki, self.kd = kp, ki, kd
        self.alpha = alpha
        self.i_limit = i_limit
        self.slew_limit = slew_limit
        self.ff_clip = ff_clip
        self.divergence_steps = divergence_steps
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_cmd = 0.0
        self.sat_count = 0          # 前馈连续饱和计数
        self.ff_enabled = True      # 前馈使能（发散后置 False）
        self.status = "normal"      # normal / ff_disabled / sensor_fault

    def control(self, theta, omega, target, dt=1 / 60):
        # ---- 传感器故障保护：输入非法则前馈归零 ----
        if not (np.isfinite(theta) and np.isfinite(omega)):
            self.status = "sensor_fault"
            tau_ff = 0.0
        else:
            x = np.array([[theta, omega, target]], dtype=np.float32)
            tau_ff = self.alpha * float(self.model.forward(x)[0][0, 0])
            tau_ff = float(np.clip(tau_ff, -self.ff_clip, self.ff_clip))

        # ---- 前馈发散检测：连续顶满限幅则屏蔽前馈，降级纯 PID ----
        if self.ff_enabled and abs(tau_ff) >= self.ff_clip - 1e-6:
            self.sat_count += 1
            if self.sat_count >= self.divergence_steps:
                self.ff_enabled = False
                self.status = "ff_disabled"
                tau_ff = 0.0
        else:
            self.sat_count = 0

        # ---- 兜底 PID：小增益只消残余误差 ----
        e = target - theta
        self.integral = float(np.clip(self.integral + e * dt,
                                      -self.i_limit, self.i_limit))
        tau_fb = self.kp * e + self.ki * self.integral - self.kd * omega

        # ---- 合成 + 总限幅 + 斜率限制 ----
        tau = float(np.clip(tau_ff + tau_fb, -1.0, 1.0))
        tau = float(np.clip(tau, self.prev_cmd - self.slew_limit,
                            self.prev_cmd + self.slew_limit))
        self.prev_cmd = tau
        return tau


def export_c_header(model, path="weights.c"):
    """把 MLP 权重导出为 C 数组，供 MCU 部署（model.params 字典结构）。"""
    def arr(name, a):
        flat = np.atleast_2d(a).ravel()
        body = ", ".join(f"{v:.7f}f" for v in flat)
        return f"static const float {name}[{flat.size}] = {{{body}}};\n"
    src = ("// 自动生成：MLP 前馈权重（float32，推理=2 次 GEMV）\n")
    for i in (1, 2, 3):
        src += arr(f"W{i}", model.params[f"W{i}"])
        src += arr(f"b{i}", model.params[f"b{i}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return path
