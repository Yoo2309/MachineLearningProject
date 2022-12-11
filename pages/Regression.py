import streamlit as st

from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.markdown("# Hồi quy ❄️")
st.sidebar.markdown("# Regression ❄️")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.selectbox('Chọn bài',['Bai01','Bai02', 'Bai03', 'Bai04', 'Bai05']) 

if (app_mode == 'Bai01'):
    st.title("Bài 01")
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
    mot = np.ones((1, 13), dtype = np.int32)
    X_bar = np.vstack((mot, X))
    X_bar_T = X_bar.T
    A = np.matmul(X_bar, X_bar_T)
    y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    b = np.matmul(X_bar, y)
    A_inv = np.linalg.pinv(A)
    w = np.matmul(A_inv, b)
    x =[]
    e = []
    for i in range(0,13):
        x.append(X[0,i])
    for i in range(0,13):
        e.append(y.T[0,i])
    df = pd.DataFrame({
        'Chiều cao(cm)':x,
        'Cân nặng tương ứng(kg)':e,
    })
    st.header('Bảng số liệu tham khảo:')
    st.write(df.T)
    
    x1 = X[0, 0]
    y1 = x1*w[1, 0] + w[0, 0]
    x2 = X[0, -1]
    y2 = x2*w[1, 0] + w[0, 0]
    plt.plot(X, y.T, 'ro')
    plt.plot([x1, x2], [y1, y2])  
    st.header('Đồ thị')  
    st.pyplot()
    # weights
    st.write('Ví dụ một số dự đoán:')
    w_0, w_1 = w[0], w[1]
    y1 = w_1*150 + w_0
    y2 = w_1*170 + w_0
    st.write('Chiều cao 150cm, cân nặng thực tế 50kg, cân nặng dự đoán %.2fkg' %(y1))
    st.write('Chiều cao 170cm, cân nặng thực tế 62kg, cân nặng dự đoán %.2fkg' %(y2))
    st.set_option('deprecation.showPyplotGlobalUse', False)
elif (app_mode == 'Bai02'):
    st.title("Bài 02")
    
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("Scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
    X = X[:,0]
    fig, ax = plt.subplots()
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    st.pyplot(fig)
elif (app_mode == 'Bai03'):
    st.title("Bài 03")
    st.write('Hồi quy bậc 2')
    
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    X2 = X**2
    # print(X)
    # print(X2)
    X_poly = np.hstack((X, X2))
    # print(X_poly)

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, y)
    st.write(lin_reg.intercept_)
    st.write(lin_reg.coef_)
    a = lin_reg.intercept_[0]
    b = lin_reg.coef_[0,0]
    c = lin_reg.coef_[0,1]
    st.write('a: ', a)
    st.write('b: ', b)
    st.write('c: ', c)

    x_ve = np.linspace(-3,3,m)
    y_ve = a + b*x_ve + c*x_ve**2
    fig, ax = plt.subplots()
    plt.plot(X, y, 'o')
    plt.plot(x_ve, y_ve, 'r')

    # Tinh sai so
    loss = 0 
    for i in range(0, m):
        y_mu = a + b*X_poly[i,0] + c*X_poly[i,1]
        sai_so = (y[i] - y_mu)**2 
        loss = loss + sai_so
    loss = loss/(2*m)
    st.write('loss = %.6f' % loss)

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('sai so binh phuong trung binh: %.6f' % (sai_so_binh_phuong_trung_binh/2))
    st.pyplot(fig)
elif (app_mode == 'Bai04'):
    st.title("Bài 04")
    st.write('Sự nhiễu của 1 cặp dữ liệu trong model (150cm, 90kg)')
    
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    
    fig, ax = plt.subplots()
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    st.pyplot(fig)
else:
    st.title("Bài 05")
    st.write('Cách khắc phục sự nhiễu ở Bài 04')
    
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    huber_reg = linear_model.HuberRegressor()
    huber_reg.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("Scikit-learn’s solution : w_1 = ", huber_reg.coef_[0], "w_0 = ", huber_reg.intercept_)
    fig, ax = plt.subplots()
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = huber_reg.coef_[0]
    b = huber_reg.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    st.pyplot(fig)