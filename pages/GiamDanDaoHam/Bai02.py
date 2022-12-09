from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt

X = np.random.rand(1000)
y = 4 + 3 * X + .5*np.random.randn(1000)

model = LinearRegression()
model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
w, b = model.coef_[0][0], model.intercept_[0]
x0 = 0
x1 = 1
y0 = w*x0 + b
y1 = w*x1 + b

plt.plot(X, y, 'bo', markersize = 2)
plt.plot([x0, x1], [y0, y1], 'r')

plt.show()

