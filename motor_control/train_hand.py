# -*- coding: utf-8 -*-
"""五指手 MLP 训练：输入 21 维（θ×7, ω×7, tgt×7），输出 7 维力矩。

网络 21→128→128→7（约 25000 参数），输出层 tanh 天然限幅 [-1,1]。
"""
import numpy as np
from model import MLP


def train(epochs=150, batch_size=4096, lr=3e-3):
    data = np.load("data/hand_data.npz")
    X, Y = data["X"], data["Y"]

    idx = np.random.default_rng(7).permutation(len(X))
    X, Y = X[idx], Y[idx]

    model = MLP(sizes=(21, 128, 128, 7))
    n = len(X)

    for epoch in range(epochs):
        total = 0.0
        for i in range(0, n, batch_size):
            xb, yb = X[i:i + batch_size], Y[i:i + batch_size]
            out, cache = model.forward(xb)
            loss = ((out - yb) ** 2).mean()
            grads = model.backward(cache, yb)
            for k in model.params:
                model.params[k] -= lr * grads[k]
            total += loss * len(xb)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs}  loss={total / n:.6f}")

    model.save("model_hand.npz")
    print("Saved model_hand.npz")


if __name__ == "__main__":
    train()
