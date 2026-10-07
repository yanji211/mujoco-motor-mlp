import mujoco
import numpy as np

TORQUE_LIMIT = 1.0  # 与 motor.xml 中 ctrlrange 一致

class MotorSim:
    def __init__(self, xml_path="motor.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def reset(self, theta=0.0, omega=0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = theta
        self.data.qvel[0] = omega
        mujoco.mj_forward(self.model, self.data)

    def step(self, torque_norm):
        # torque_norm 归一化到 [-1, 1]，实际力矩 = torque_norm * TORQUE_LIMIT
        u = np.clip(torque_norm, -1.0, 1.0)
        self.data.ctrl[0] = u * TORQUE_LIMIT
        mujoco.mj_step(self.model, self.data)
        return self.data.qpos[0], self.data.qvel[0]

    def get_state(self):
        return self.data.qpos[0], self.data.qvel[0]
