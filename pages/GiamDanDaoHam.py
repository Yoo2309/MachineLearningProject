from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

import streamlit as st
st.markdown("# Giảm dần đạo hàm ❄️")
st.sidebar.markdown("# Giảm dần đạo hàm ❄️")

app_mode = st.sidebar.selectbox('Select Page',['Bai01','Bai02', 'Bai02a','Bai03', 'Bai04', 'Bai05', 'Temp']) 

if (app_mode == 'Bai01'):
    st.title('Bài 1')
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
    
    x0 = -5
    eta = 0.1
    (x, it) = myGD1(x0, eta)
    x = np.array(x)
    y = cost(x)

    n = 101
    xx = np.linspace(-6, 6, n)
    yy = xx**2 + 5*np.sin(xx)

    fig, ax = plt.subplots()
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
    st.write(fig)
elif (app_mode == 'Bai02'):
    st.title('Bài 2')
    
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    x0 = 0
    x1 = 1
    y0 = w*x0 + b
    y1 = w*x1 + b

    fig, ax = plt.subplots()
    plt.plot(X, y, 'bo', markersize = 2)
    plt.plot([x0, x1], [y0, y1], 'r')
    st.pyplot(fig)    
elif (app_mode == 'Bai02a'):
    st.title('Bài 02a')
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000) # noise added

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    w, b = model.coef_[0][0], model.intercept_[0]
    sol_sklearn = np.array([b, w])
    st.write('Solution found by sklearn:', sol_sklearn)

    # Building Xbar 
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
    st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
elif (app_mode == 'Bai03'):
    st.title('Bài 3')
    np.random.seed(100)
    N = 1000
    X = np.random.rand(N)
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    st.write('b = %.4f va w = %.4f' % (b, w))

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
    st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    # for item in w1:
    #     st.write(item, cost(item))

    # st.write(len(w1))

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
        
    # Create an object for graph layout
    data = go.Surface(x = b, y = w, z = z)
    fig = go.Figure(data)

    st.plotly_chart(fig)
    

elif (app_mode == 'Bai04'):
    st.title('Bài 4')
    x = np.linspace(-2, 2, 21)
    y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    fig, ax = plt.subplots()
    plt.contour(X, Y, Z, 10)
    st.pyplot(fig)
elif (app_mode == 'Bai05'):
    st.title('Bài 5')
    np.random.seed(100)
    N = 1000
    X = np.random.rand(N)
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    st.write('b = %.4f va w = %.4f' % (b, w))

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
    st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    # for item in w1:
    #     st.write(item, cost(item))

    # st.write(len(w1))

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

    fig, ax = plt.subplots()
    plt.contour(b, w, z, 45)
    bdata = []
    wdata = []
    for item in w1:
        plt.plot(item[0], item[1], 'ro', markersize = 3)
        bdata.append(item[0])
        wdata.append(item[1])

    plt.plot(bdata, wdata, color = 'b')

    plt.xlabel('b')
    plt.ylabel('w')
    plt.axis('square')
    st.pyplot(fig)
else:
    st.title('Temp')

    X = np.linspace(-2, 2, 21)
    Y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(X, Y)
    Z = X*X + Y*Y

    # Create an object for graph layout
    data = go.Surface(x = X, y = Y, z = Z)
    fig = go.Figure(data)

    st.plotly_chart(fig)