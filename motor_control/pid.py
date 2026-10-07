class PID:
    """输出归一化力矩 [-1, 1]，实际力矩由 sim 乘以 TORQUE_LIMIT。"""
    def __init__(self, kp=20.0, ki=0.1, kd=4.0, i_limit=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def control(self, theta, omega, target, dt=0.01):
        error = target - theta
        self.integral += error * dt
        # 抗积分饱和
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        torque = self.kp * error + self.ki * self.integral - self.kd * omega
        return max(-1.0, min(1.0, torque))
