from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

class Hinge_Loss:
    def __init__(self, tradeoff=1e-2):
        self.tradeoff = tradeoff

    def __call__(self, pred, y, x, w):
        loss = max(0, 1 - y * pred) + self.tradeoff * linalg.norm(w)**2
        if y * pred >= 1:
            dLdw = 2 * self.tradeoff * w
            dLdb = 0
        else:
            dLdw = -np.dot(y, x) + 2 * self.tradeoff * w
            dLdb = -y
        return loss, dLdw, dLdb
           

class Linear_SVM:
    def __init__(self, learning_rate=1e-3, epochs=100):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.epochs = epochs
        self.loss = Hinge_Loss()

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for epoch in range(self.epochs):
            for i, x in enumerate(X):
                pred = np.dot(self.w, x) + self.b
                loss, dLdw, dLdb = self.loss(pred, y[i], x, self.w)
                self.w -= self.lr * dLdw
                self.b -= self.lr * dLdb
        return self.w, self.b

X, y = datasets.make_blobs()
y = np.where(y <= 0, -1, 1)
model = Linear_SVM()
w, b = model.fit(X, y)

def get_hyperplane(x, w, b):
    return (-w[0] * x - b) / w[1]

plt.scatter(X[:,0], X[:,1], marker='o',c=y)
min_x1 = np.min(X[:, 0])
min_x2_pred = get_hyperplane(min_x1, w, b)

max_x1 = np.max(X[:, 0])
max_x2_pred = get_hyperplane(max_x1, w, b)

plt.plot([min_x1, max_x1], [min_x2_pred, max_x2_pred])
plt.show()
