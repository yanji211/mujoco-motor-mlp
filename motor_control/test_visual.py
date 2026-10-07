import mujoco
import mujoco.viewer
import numpy as np
import time
from model import MLP

model_mlp = MLP(sizes=(3, 32, 32, 1))
model_mlp.load("model_motor.npz")

model = mujoco.MjModel.from_xml_path("motor.xml")
data = mujoco.MjData(model)

# 初始状态
data.qpos[0] = 0.0
data.qvel[0] = 0.0
target = 1.0  # 目标角度

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_duration = model.opt.timestep
    while viewer.is_running():
        step_start = time.time()
        theta = data.qpos[0]
        omega = data.qvel[0]
        x = np.array([[theta, omega, target]], dtype=np.float32)
        out, _ = model_mlp.forward(x)
        torque = float(out[0, 0])
        data.ctrl[0] = torque
        mujoco.mj_step(model, data)
        viewer.sync()
        # 实时节流，让出 CPU 给窗口事件循环，保证鼠标拖动响应
        time.sleep(max(0.0, step_start + step_duration - time.time()))
