# -*- coding: utf-8 -*-
"""灵巧手混合控制器：MLP 前馈（出力矩大头，快、平滑） + 小增益 PID 残差（消稳态误差）。

纯 MLP 行为克隆在"目标角为 0、无重力负载"的区域会带输出偏置 -> 稳态偏移；
小增益 PID 用积分项把残差清零（和 motor 项目的 hybrid.py 同一思路）。
前馈发散时 watchdog 自动降级纯 PID，保证机器不停。
"""
import numpy as np
from model import MLP
from dex_common import NJ, DT, GEAR, KP, KD, KI


class HybridDex:
    def __init__(self, model_path="model_dex.npz", alpha=1.0, gain=1.0):
        self.net = MLP(sizes=(42, 256, 256, 14))
        self.net.load(model_path)
        self.alpha = alpha
        self.KP = gain * KP
        self.KI = gain * KI
        self.KD = gain * KD
        self.integral = np.zeros(NJ)
        self._bad_steps = 0

    def reset(self):
        self.integral[:] = 0
        self._bad_steps = 0

    def control(self, th, om, tgt):
        e = tgt - th
        # 前馈
        x = np.concatenate([th, om, tgt]).astype(np.float32).reshape(1, -1)
        ff = self.net.forward(x)[0][0]
        # 残差 PID
        self.integral = np.clip(self.integral + e * DT, -5, 5)
        res = (self.KP * e + self.KI * self.integral - self.KD * om) / GEAR
        u = np.clip(self.alpha * ff + res, -1, 1)

        # watchdog：前馈长时间发散（控制饱和且误差仍大）-> 降级纯 PID
        if np.any(np.abs(ff) > 0.98) and np.max(np.abs(e)) > 0.5:
            self._bad_steps += 1
        else:
            self._bad_steps = 0
        if self._bad_steps > 30:
            u = np.clip(res / self.alpha if self.alpha > 0 else res, -1, 1)
        return u
