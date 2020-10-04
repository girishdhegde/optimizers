import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pickle 

# To be added: momentum, nestrov, RMSProp, Adam
class loss:
    def __init__(self, w=0, b=0):
        if not w:
            self.w = np.random.random(1)
        else:
            self.w = w
        if not b:
            self.b = np.random.random(1)
        else:
            self.b = b
        self.gradB = 0
        self.gradW = 0
        self.velocityW = 0
        self.velocityB = 0
        self.loss_grad = 0


    def forward(self, x):
        self.x    = x
        self.pred = x * self.w + self.b
        return self.pred

    def MSELoss(self, pred, target):
        self.loss_grad = 2 * (pred - target)
        squared_error  = (pred - target) ** 2
        return squared_error, np.mean(squared_error)

    def backward(self):
        self.gradB += np.mean(self.loss_grad)
        self.gradW  = self.gradW + np.mean(self.loss_grad * self.x)
        return self.loss_grad * self.w

    def zero_grad(self):
        self.gradB = 0
        self.gradW = 0
        self.loss_grad = 0


    def SGD(self, lr=0.1, lmda=0.0):
        self.b -= lr * self.gradB
        self.w  = self.w - lr * self.gradW - lmda * self.w
        return [self.w, self.b]

    def momentumGD(self, mu=0.9, lr=0.1, lmda=0.0):
        self.velocityW = mu * self.velocityW - lr * self.gradW
        self.velocityB = mu * self.velocityB - lr * self.gradB
        self.b += self.velocityB
        self.w  = self.w + self.velocityW - lmda * self.w
        return [self.w, self.b]
# data generation
fx = lambda x: 3 * x - 2 + np.random.uniform(-2, 2)
data = np.random.random(size=100) * 10 - 5
target = np.array(list(map(fx, data)))
# plt.scatter(data, target)
# plt.show()

l    = loss()
l.w  = -17
l.b  = 40

samples = 100
w       = np.linspace(-20, 20, samples)
b       = np.linspace(-50, 50, samples)
x, y    = np.meshgrid(w, b)
z       = np.zeros_like(x)

fig = plt.figure()
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122, projection='3d')

plt.ion()
minloss = float('inf')
minw    = 0
minb    = 0
for i in range(samples):
    for j in range(samples):
        l.w = x[i, j]
        l.b = y[i, j]
        l.zero_grad()
        out = l.forward(data)
        se, mse = l.MSELoss(out, target)
        if mse < minloss:
            minloss = mse
            minw = l.w
            minb = l.b
        z[i, j] = mse


l.w  = -17
l.b  = 40

ax1.set_xlabel('weight')
ax1.set_ylabel('bias')
ax1.set_zlabel('loss')
ax1.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma', linewidth=1, alpha=.6)



ax0.set_xlabel('x')
ax0.set_ylabel('y')

fig.suptitle('momentum gradient descent')

minx, maxx = np.min(data), np.max(data)

bs = 20
lr = 1e-2
iterations = len(data) // bs

# l.w  = -17
# l.b  = 40

zm = l.forward(data)
zm = l.MSELoss(zm, target)[1]
# ax1.legend(loc='best')
# ax1.scatter(l.w, l.b, zm, c='red', alpha=1, s=0.5, label='batch gd')

mean0 = zm
mean1 = zm
for i in range(100):
    ax1.title.set_text(f'Loss curve, epoch: {i+1}')
    # batch gradient descent
    l.zero_grad()
    out = l.forward(data)
    se, mse0 = l.MSELoss(out, target)
    mean0 = mse0

    ax1.scatter(l.w, l.b, mean0, c='red', alpha=0.5, s=2)
    if i > 0:
        ax1.plot([prevw, l.w], [prevb, l.b], [prevz, mean0], alpha=0.3, c='red')
    prevw = l.w
    prevb = l.b
    prevz = mean0

    l.backward()
    l.momentumGD(lr=lr)

    ax0.clear()
    ax0.title.set_text(f'prediction, epoch: {i+1}')
    ax0.scatter(data, target)
    ax0.plot([minx, maxx], [l.forward(minx), l.forward(maxx)], color='red')

    plt.draw()
    plt.pause(2e-1)
    print(i)
