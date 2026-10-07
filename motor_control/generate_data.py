import numpy as np
from sim import MotorSim
from pid import PID

def generate(n_episodes=2000, steps_per_ep=200, save_path="data/motor_data.npz"):
    np.random.seed(42)  # 固定种子保证可复现
    sim = MotorSim()
    pid = PID()

    X, Y = [], []

    for ep in range(n_episodes):
        theta = np.random.uniform(-1.5, 1.5)
        omega = 0.0
        target = np.random.uniform(-1.5, 1.5)
        sim.reset(theta, omega)
        pid.reset()

        for _ in range(steps_per_ep):
            theta, omega = sim.get_state()
            torque = pid.control(theta, omega, target)
            X.append([theta, omega, target])
            Y.append([torque])
            sim.step(torque)

        if (ep + 1) % 200 == 0:
            print(f"episode {ep+1}/{n_episodes}")

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    np.savez(save_path, X=X, Y=Y)
    print(f"Saved {X.shape[0]} samples to {save_path}")

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate()
