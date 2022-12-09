from matplotlib import pyplot as plt
import numpy as np

def grad(x):
    return 2*x+ 5*np.cos(x)
def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3: # just a small number
            break
        x.append(x_new)
    return (x, it)

if __name__ == "__main__":
    x0 = -5
    eta = 0.1
    (x, it) = myGD1(x0, eta)
    x = np.array(x)
    y = cost(x)

    n = 101
    xx = np.linspace(-6, 6, n)
    yy = xx**2 + 5*np.sin(xx)

    plt.subplot(2,4,1)
    plt.plot(xx, yy)
    index = 0
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,2)
    plt.plot(xx, yy)
    index = 1
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,3)
    plt.plot(xx, yy)
    index = 2
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,4)
    plt.plot(xx, yy)
    index = 3
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,5)
    plt.plot(xx, yy)
    index = 4
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,6)
    plt.plot(xx, yy)
    index = 5
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,7)
    plt.plot(xx, yy)
    index = 7
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,8)
    plt.plot(xx, yy)
    index = 11
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.tight_layout()
    plt.show()


