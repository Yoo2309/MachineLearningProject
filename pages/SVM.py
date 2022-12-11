import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt
import numpy as np

st.markdown("# Support Vector Machine ❄️")
st.sidebar.markdown("# SVM ❄️")

html = '''
    <p /align="justify";/>SVM là một phương pháp hiệu quả cho bài toán phân lớp dữ liệu. Nó là một công cụ đắc lực cho các bài toán về xử lý ảnh, phân loại văn bản, phân tích quan điểm. Một yếu tố làm nên hiệu quả của SVM đó là việc sử dụng Kernel function khiến cho các phương pháp chuyển không gian trở nên linh hoạt hơn.</p>
    '''
st.markdown(html, unsafe_allow_html=True)
st.image('images/svm.png')

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.selectbox('Chọn bài',['Bai1','Bai1a','Bai2','plot_linearsvc_support_vectors']) 

if app_mode=='Bai1':
    st.title("Bài 1") 

    np.random.seed(100)
    N = 150

    centers = [[2, 2], [7, 7]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 

    nhom_0 = []
    nhom_1 = []

    SIZE = train_data.shape[0]
    for i in range(0, SIZE):
        if train_labels[i] == 0:
            nhom_0.append([train_data[i,0], train_data[i,1]])
        elif train_labels[i] == 1:
            nhom_1.append([train_data[i,0], train_data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    col1, col2 = st.columns(2)
    with col1:
        st.write('Nhóm 0', nhom_0, use_column_width=True)
    with col2:
        st.write('Nhóm 1', nhom_1, use_column_width=True)

    svc = LinearSVC(C = 100, loss="hinge", random_state=42, max_iter = 100000)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    predicted = svc.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write('Sai số:', sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)

    st.write('Kết quả nhận dạng là nhóm:', ket_qua[0])

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]

    plt.plot(xx, yy, 'b')


    decision_function = svc.decision_function(train_data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = train_data[support_vector_indices]
    support_vectors_x = support_vectors[:,0]
    support_vectors_y = support_vectors[:,1]

    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(
        svc,
        train_data,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.legend(['Nhóm 0', 'Nhóm 1'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig=None, clear_figure=True)     
    
elif app_mode == 'Bai1a':
    st.title('Bài 1a')

    np.random.seed(100)
    N = 150

    centers = [[2, 2], [7, 7]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 

    nhom_0 = []
    nhom_1 = []

    SIZE = train_data.shape[0]
    for i in range(0, SIZE):
        if train_labels[i] == 0:
            nhom_0.append([train_data[i,0], train_data[i,1]])
        elif train_labels[i] == 1:
            nhom_1.append([train_data[i,0], train_data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    col1, col2 = st.columns(2)
    with col1:
        st.write('Nhóm 0', nhom_0, use_column_width=True)
    with col2:
        st.write('Nhóm 1', nhom_1, use_column_width=True)




    svc = LinearSVC(C = 100, loss="hinge", random_state=42, max_iter = 100000)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    predicted = svc.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write('sai số:', sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)

    st.write('Kết quả nhận dạng là nhóm:', ket_qua[0])

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]

    plt.plot(xx, yy, 'b')

    w = he_so[0]
    a = w[0]
    b = w[1]
    c = intercept[0]
    
    distance = np.zeros(SIZE, np.float32)
    for i in range(0, SIZE):
        x0 = train_data[i,0]
        y0 = train_data[i,1]
        d = np.abs(a*x0 + b*y0 + c)/np.sqrt(a**2 + b**2)
        distance[i] = d
    st.write('Khoảng cách:')
    st.write(distance)
    vi_tri_min = np.argmin(distance)
    min_val = np.min(distance)
    st.write('vị trí min', vi_tri_min)
    st.write('giá trị min', min_val)
    st.write('Những giá trị gần min')
    vi_tri = []
    for i in range(0, SIZE):
        if (distance[i] - min_val) <= 1.0E-3:
            st.write(distance[i])
            vi_tri.append(i)
    st.write(vi_tri)
    for i in vi_tri:
        x = train_data[i,0]
        y = train_data[i,1]
        plt.plot(x, y, 'rs')

    i = vi_tri[0]
    x0 = train_data[i,0]
    y0 = train_data[i,1]
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')

    i = vi_tri[2]
    x0 = train_data[i,0]
    y0 = train_data[i,1]
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')


    plt.legend(['Nhom 0', 'Nhom 1'])

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig=None, clear_figure=True) 
    
elif app_mode == 'Bai2':
    st.title("Bài 2") 

    np.random.seed(100)
    N = 150

    centers = [[2, 2], [7, 7]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 

    nhom_0 = []
    nhom_1 = []

    SIZE = train_data.shape[0]
    for i in range(0, SIZE):
        if train_labels[i] == 0:
            nhom_0.append([train_data[i,0], train_data[i,1]])
        elif train_labels[i] == 1:
            nhom_1.append([train_data[i,0], train_data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    col1, col2 = st.columns(2)
    with col1:
        st.write('Nhóm 0', nhom_0, use_column_width=True)
    with col2:
        st.write('Nhóm 1', nhom_1, use_column_width=True)


    svc = SVC(C = 100, kernel='linear', random_state=42)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    predicted = svc.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write('sai số:', sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)

    st.write('Kết quả nhận dạng là nhóm:', ket_qua[0])

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]
    plt.plot(xx, yy, 'b')

    support_vectors = svc.support_vectors_
    st.write('support vector: ', support_vectors)

    w = he_so[0]
    a = w[0]
    b = w[1]

    i = 0
    x0 = support_vectors[i,0]
    y0 = support_vectors[i,1]
    plt.plot(x0, y0, 'rs')
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')

    i = 1
    x0 = support_vectors[i,0]
    y0 = support_vectors[i,1]
    plt.plot(x0, y0, 'rs')
    c = -a*x0 -b*y0
    xx = np.linspace(2, 7, 100)
    yy = -a*xx/b - c/b
    plt.plot(xx, yy, 'b--')

    plt.legend(['Nhom 0', 'Nhom 1'])

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig=None, clear_figure=True) 

elif app_mode == 'plot_linearsvc_support_vectors':
    st.title('Plot linear SVC support vectors')
    X, y = make_blobs(n_samples=40, centers=2, random_state=0)

    plt.figure(figsize=(10, 5))
    # "hinge" is the standard SVM loss
    clf = LinearSVC(C=100, loss="hinge", random_state=42).fit(X, y)
    # obtain the support vectors through the decision function
    decision_function = clf.decision_function(X)
    # we can also calculate the decision function manually
    # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # The support vectors are the samples that lie within the margin
    # boundaries, whose size is conventionally constrained to 1
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = X[support_vector_indices]

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title("C = 100")

    fig = plt.tight_layout()

    st.pyplot(fig) 