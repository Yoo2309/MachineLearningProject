from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


np.random.seed(100)
N = 1000
X = np.random.rand(N)
y = 4 + 3 * X + .5*np.random.randn(N)

model = LinearRegression()
model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
w, b = model.coef_[0][0], model.intercept_[0]
print('b = %.4f va w = %.4f' % (b, w))

one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

def myGD(w_init, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it)

w_init = np.array([0, 0])
(w1, it1) = myGD(w_init, 1)
print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
for item in w1:
    print(item, cost(item))

print(len(w1))

A = N/(2*N)
B = np.sum(X*X)/(2*N)
C = -np.sum(y)/(2*N)
D = -np.sum(X*y)/(2*N)
E = np.sum(X)/(2*N)
F = np.sum(y*y)/(2*N)

b = np.linspace(0,6,21)
w = np.linspace(0,6,21)
b, w = np.meshgrid(b, w)
z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F

temp = w1[0]
bb = temp[0]
ww = temp[1]
zz = cost(temp) 
ax = plt.axes(projection="3d")
ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

temp = w1[1]
bb = temp[0]
ww = temp[1]
zz = cost(temp) 
ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

temp = w1[2]
bb = temp[0]
ww = temp[1]
zz = cost(temp) 
ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

temp = w1[3]
bb = temp[0]
ww = temp[1]
zz = cost(temp) 
ax.plot3D(bb, ww, zz, 'ro', markersize = 3)


ax.plot_wireframe(b, w, z)
ax.set_xlabel("b")
ax.set_ylabel("w")
plt.show()

