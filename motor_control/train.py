import numpy as np
from model import MLP

def train(epochs=300, batch_size=256, lr=3e-3):
    data = np.load("data/motor_data.npz")
    X, Y = data["X"], data["Y"]

    # 打乱
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    model = MLP(sizes=(3, 32, 32, 1))
    n = len(X)

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]
            out, cache = model.forward(xb)
            loss = ((out - yb) ** 2).mean()
            grads = model.backward(cache, yb)
            for k in model.params:
                model.params[k] -= lr * grads[k]
            total_loss += loss * len(xb)
        print(f"epoch {epoch+1}/{epochs}  loss={total_loss/n:.6f}")

    model.save("model_motor.npz")
    print("Saved model_motor.npz")

if __name__ == "__main__":
    train()
