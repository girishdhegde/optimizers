import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.animation import FuncAnimation


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


l    = loss()
l.w  = -17
l.b  = 40

samples = 100
w       = np.linspace(-20, 20, samples)
b       = np.linspace(-50, 50, samples)
x, y    = np.meshgrid(w, b)
z       = np.zeros_like(x)

fig = plt.figure(figsize=(20, 10))

ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122, projection='3d')
fig.suptitle('Momentum Gradient Descent')

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



ax1.set_xlabel('weight')
ax1.set_ylabel('bias')
ax1.set_zlabel('loss')
ax1.view_init(elev=35)
ax1.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma', linewidth=1, alpha=.6)



ax0.set_xlabel('x')
ax0.set_ylabel('y')

minx, maxx = np.min(data), np.max(data)
miny, maxy = np.min(target), np.max(target)

bs = 20
lr = 2e-3
iterations = len(data) // bs

l.w  = -17
l.b  = 40

zm = l.forward(data)
zm = l.MSELoss(zm, target)[1]

mean0 = zm
mean1 = zm
out = l.forward(data)
se, mse0 = l.MSELoss(out, target)
prevb, prevw, prevz = l.b, l.w, mse0
def update(i):
    global prevb, prevw, prevz
    ax1.title.set_text(f'Loss, epoch: {i+1}')
    l.zero_grad()
    out = l.forward(data)
    se, mse0 = l.MSELoss(out, target)
    mean0 = mse0

    ax1.scatter(l.w, l.b, mean0, c='red', alpha=0.5, s=2)
    ax1.scatter(minw, minb, minloss, marker='+', s=60)
    ax1.plot([prevw, l.w], [prevb, l.b], [prevz, mean0], alpha=0.3, c='red')    

    prevw = l.w
    prevb = l.b
    prevz = mean0

    l.backward()
    l.momentumGD(lr=lr)

    ax0.clear()
    ax0.title.set_text(f'prediction, epoch: {i+1}')
    ax0.scatter(data, target, label='data')
    ax0.plot([minx, maxx], [l.forward(minx), l.forward(maxx)], color='red', label='model')
    ax0.legend()
    ax0.set_ylim(miny, maxy)

    plt.draw()
    plt.pause(2e-12)
    print(f'epoch: {i}')
    return fig


anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200)
anim.save('momentum.gif', dpi=1, writer='imagemagick')
