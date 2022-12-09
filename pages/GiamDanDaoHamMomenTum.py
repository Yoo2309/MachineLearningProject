import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

st.markdown("# Giảm dần đạo hàm MomenTum ❄️")
st.sidebar.markdown("# GiamDanDaoHamMomenTum ❄️")

x = np.linspace(-5, 5, 100)
y = x**2 + 10*np.sin(x)
fig, ax = plt.subplots()
plt.plot(x, y)
x_1 = -3.5
y_1 = x_1**2 + 10*np.sin(x_1)

m = 2*x_1 + 10*np.cos(x_1)
dx = 1
dy = m*dx
L = np.sqrt(dx**2 + dy**2)
he_so = 5
dx = he_so*dx / L
dy = he_so*dy / L

plt.arrow(x_1 + 0.5 , y_1, dx, dy, head_width = 0.5)

plt.plot(x_1 + 0.5, y_1, 'ro', markersize = 20)

x_2 = 0
y_2 = x_2**2 + 10*np.sin(x_2)

m = 2*x_2 + 10*np.cos(x_2)
dx = -1
dy = m*dx
L = np.sqrt(dx**2 + dy**2)
he_so = 5
dx = he_so*dx / L
dy = he_so*dy / L

plt.arrow(x_2, y_2 + 4, dx, dy, head_width = 0.5)
plt.plot(x_2, y_2 + 4, 'yo', markersize = 20)

plt.fill_between(x, y, -10)

plt.axis([-6, 6, -10, 40])
st.pyplot(fig)