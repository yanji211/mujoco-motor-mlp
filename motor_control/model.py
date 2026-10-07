import numpy as np

class MLP:
    def __init__(self, sizes=(3, 32, 32, 1)):
        self.sizes = sizes
        self.params = {}
        for i in range(len(sizes) - 1):
            self.params[f"W{i+1}"] = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            self.params[f"b{i+1}"] = np.zeros((1, sizes[i+1]))

    def forward(self, x):
        cache = {"a0": x}
        a = x
        L = len(self.sizes) - 1
        for i in range(1, L):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            a = np.maximum(0, z)  # ReLU
            cache[f"z{i}"] = z
            cache[f"a{i}"] = a
        z = a @ self.params[f"W{L}"] + self.params[f"b{L}"]
        out = np.tanh(z)  # 输出限制在 -1~1
        cache[f"z{L}"] = z
        cache[f"a{L}"] = out
        return out, cache

    def backward(self, cache, y):
        L = len(self.sizes) - 1
        m = y.shape[0]
        grads = {}
        # 输出层：MSE + tanh
        out = cache[f"a{L}"]
        dout = 2 * (out - y) / m
        dz = dout * (1 - out ** 2)  # tanh 导数
        grads[f"W{L}"] = cache[f"a{L-1}"].T @ dz
        grads[f"b{L}"] = dz.sum(axis=0, keepdims=True)
        da = dz @ self.params[f"W{L}"].T

        for i in range(L - 1, 0, -1):
            dz = da * (cache[f"z{i}"] > 0)  # ReLU 导数
            grads[f"W{i}"] = cache[f"a{i-1}"].T @ dz
            grads[f"b{i}"] = dz.sum(axis=0, keepdims=True)
            if i > 1:
                da = dz @ self.params[f"W{i}"].T
        return grads

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        data = np.load(path)
        for k in data.files:
            self.params[k] = data[k]
