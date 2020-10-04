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
        self.t     = 0
        self.mW    = 0
        self.mB    = 0
        self.vW    = 0
        self.vB    = 0
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

    def rmsProp(self, lr=0.1, rho=0.9, eps=1e-8, lmda=0.0):
        self.velocityW = rho * self.velocityW + (1 - rho) * (self.gradW ** 2)
        self.velocityB = rho * self.velocityB + (1 - rho) * (self.gradB ** 2)
        self.b  = self.b - (lr/(self.velocityB ** 0.5 + eps)) * self.gradB
        self.w  = self.w - (lr/(self.velocityW ** 0.5 + eps)) * self.gradW - lmda * self.w
        return [self.w, self.b]
   
    def adam(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.0):
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * self.gradW
        self.mB = beta1 * self.mB + (1 - beta1) * self.gradB
        self.vW = beta2 * self.vW + (1 - beta2) * (self.gradW ** 2)
        self.vB = beta2 * self.vB + (1 - beta2) * (self.gradB ** 2)

        # mW      = self.mW / (1 - (beta1 ** self.t))
        # mB      = self.mB / (1 - (beta1 ** self.t))
        # vW      = self.vW / (1 - (beta2 ** self.t))
        # vB      = self.vB / (1 - (beta2 ** self.t))

        self.b  = self.b - (lr * self.mB / (self.vB ** 0.5 + eps))
        self.w  = self.w - (lr * self.mW / (self.vW ** 0.5 + eps))
        return [self.w, self.b]

  # data generation
fx = lambda x: 3 * x - 2 + np.random.uniform(-2, 2)
data = np.random.random(size=100) * 10 - 5
target = np.array(list(map(fx, data)))


l_sgd    = loss()
l_mom    = loss()
l_rms    = loss()
l_adm    = loss()

l_sgd.w  = -17
l_mom.w  = -17
l_rms.w  = -17
l_adm.w  = -17
l_sgd.b  = 40
l_mom.b  = 40
l_rms.b  = 40
l_adm.b  = 40

samples = 100
w       = np.linspace(-20, 20, samples)
b       = np.linspace(-50, 50, samples)
x, y    = np.meshgrid(w, b)
z       = np.zeros_like(x)

fig = plt.figure(figsize=(20, 10))

ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122, projection='3d')
fig.suptitle('Optimizers')

plt.ion()
minloss = float('inf')
minw    = 0
minb    = 0
for i in range(samples):
    for j in range(samples):
        l_sgd.w = x[i, j]
        l_sgd.b = y[i, j]
        l_sgd.zero_grad()
        out = l_sgd.forward(data)
        se, mse = l_sgd.MSELoss(out, target)
        if mse < minloss:
            minloss = mse
            minw = l_sgd.w
            minb = l_sgd.b
        z[i, j] = mse


ax1.set_xlabel('weight')
ax1.set_ylabel('bias')
ax1.set_zlabel('loss')
ax1.view_init(elev=37)
ax1.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma', linewidth=1, alpha=.6)

ax0.set_xlabel('epoch')
ax0.set_ylabel('loss')

minx, maxx = np.min(data), np.max(data)
miny, maxy = np.min(target), np.max(target)

bs  = 20
lr1 = 1e-2
lr2 = 2e-3
lr3 = 1e0
lr4 = 3e-1
iterations = len(data) // bs

l_sgd.w  = -17
l_mom.w  = -17
l_rms.w  = -17
l_adm.w  = -17
l_sgd.b  = 40
l_mom.b  = 40
l_rms.b  = 40
l_adm.b  = 40

zm = l_sgd.forward(data)
zm = l_sgd.MSELoss(zm, target)[1]

mse1 = zm
mse2 = zm
mse3 = zm
mse4 = zm

prevb1, prevw1, prevz1 = l_sgd.b, l_sgd.w, zm
prevb2, prevw2, prevz2 = l_mom.b, l_mom.w, zm
prevb3, prevw3, prevz3 = l_rms.b, l_rms.w, zm
prevb4, prevw4, prevz4 = l_adm.b, l_adm.w, zm

ax1.scatter(minw, minb, minloss, color='blue', s=5, label=f'SGD(lr={lr1})')
ax1.scatter(minw, minb, minloss, color='black', s=5, label=f'Momentum(lr={lr2})')
ax1.scatter(minw, minb, minloss, color='green', s=5, label=f'RMSProp(lr={lr3})')
ax1.scatter(minw, minb, minloss, color='red', s=5, label=f'Adam(lr={lr4})')
ax0.scatter(0, 0, color='blue', s=5, label=f'SGD(lr={lr1})')
ax0.scatter(0, 0, color='black', s=5, label=f'Momentum(lr={lr2})')
ax0.scatter(0, 0, color='green', s=5, label=f'RMSProp(lr={lr3})')
ax0.scatter(0, 0, color='red', s=5, label=f'Adam(lr={lr4})')

ax1.scatter(minw, minb, minloss, marker='+', s=60)

def update(i):
    global prevb1, prevw1, prevz1, prevb2, prevw2, prevz2, \
           prevb3, prevw3, prevz3, prevb4, prevw4, prevz4
   
    ax1.title.set_text(f'Loss vs Parameters, epoch: {i+1}')

    l_sgd.zero_grad()
    l_mom.zero_grad()
    l_rms.zero_grad()
    l_adm.zero_grad()

    out1 = l_sgd.forward(data)
    out2 = l_mom.forward(data)
    out3 = l_rms.forward(data)
    out4 = l_adm.forward(data)

    se1, mse1 = l_sgd.MSELoss(out1, target)
    se2, mse2 = l_mom.MSELoss(out2, target)
    se3, mse3 = l_rms.MSELoss(out3, target)
    se4, mse4 = l_adm.MSELoss(out4, target)

    ax1.scatter(l_sgd.w, l_sgd.b, mse1, c='blue', alpha=0.5, s=2)
    ax1.scatter(l_mom.w, l_mom.b, mse2, c='black', alpha=0.5, s=2)
    ax1.scatter(l_rms.w, l_rms.b, mse3, c='green', alpha=0.5, s=2)
    ax1.scatter(l_adm.w, l_adm.b, mse4, c='red', alpha=0.5, s=2)
    
    ax1.plot([prevw1, l_sgd.w], [prevb1, l_sgd.b], [prevz1, mse1], alpha=0.3, c='blue')    
    ax1.plot([prevw2, l_mom.w], [prevb2, l_mom.b], [prevz2, mse2], alpha=0.3, c='black')    
    ax1.plot([prevw3, l_rms.w], [prevb3, l_rms.b], [prevz3, mse3], alpha=0.3, c='green')    
    ax1.plot([prevw4, l_adm.w], [prevb4, l_adm.b], [prevz4, mse4], alpha=0.3, c='red')    
    ax1.legend()

    ax0.title.set_text(f'Loss vs Epoch, epoch: {i+1}')
    if i > 0:
        ax0.plot([i-1, i], [prevz1, mse1], color='blue')
        ax0.plot([i-1, i], [prevz2, mse2], color='black')
        ax0.plot([i-1, i], [prevz3, mse3], color='green')
        ax0.plot([i-1, i], [prevz4, mse4], color='red')
    ax0.legend()

    prevw1, prevb1, prevz1 = l_sgd.w, l_sgd.b, mse1
    prevw2, prevb2, prevz2 = l_mom.w, l_mom.b, mse2
    prevw3, prevb3, prevz3 = l_rms.w, l_rms.b, mse3
    prevw4, prevb4, prevz4 = l_adm.w, l_adm.b, mse4

    l_sgd.backward()
    l_mom.backward()
    l_rms.backward()
    l_adm.backward()

    l_sgd.SGD(lr=lr1)
    l_mom.momentumGD(lr=lr2)
    l_rms.rmsProp(lr=lr3)
    l_adm.adam(lr=lr4)

    plt.draw()
    plt.pause(2e-12)
    print(f'epoch: {i}')


    return fig


anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200)
anim.save('optimizers.gif', dpi=1, writer='imagemagick')
# for i in range(100):
#     update(i)