# -*- coding: utf-8 -*-
"""灵巧手 MLP 训练：输入 42 维（θ×14, ω×14, tgt×14），输出 14 维力矩。

网络 42→256→256→14（约 8 万参数），输出层 tanh 天然限幅 [-1,1]。
"""
import numpy as np
from model import MLP


def train(epochs=120, batch_size=4096, lr=3e-3):
    data = np.load("data/dex_data.npz")
    X, Y = data["X"], data["Y"]

    idx = np.random.default_rng(7).permutation(len(X))
    X, Y = X[idx], Y[idx]

    model = MLP(sizes=(42, 256, 256, 14))
    n = len(X)

    for epoch in range(epochs):
        total = 0.0
        perm = np.random.default_rng(epoch).permutation(n)
        Xs, Ys = X[perm], Y[perm]
        for i in range(0, n, batch_size):
            xb, yb = Xs[i:i + batch_size], Ys[i:i + batch_size]
            out, cache = model.forward(xb)
            loss = ((out - yb) ** 2).mean()
            grads = model.backward(cache, yb)
            for k in model.params:
                model.params[k] -= lr * grads[k]
            total += loss * len(xb)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs}  loss={total / n:.6f}")

    model.save("model_dex.npz")
    print("Saved model_dex.npz")


if __name__ == "__main__":
    train()
