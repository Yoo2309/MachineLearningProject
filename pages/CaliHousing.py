import streamlit as st

st.markdown("# Cali Housing ❄️")
st.sidebar.markdown("# Cali Housing ❄️")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Decision_Tree_Regression','Linear_Regression','Random_Forest_Regression_Grid_Search_CV',
                                                'Random_Forest_Regression_Random_Search_CV', 'Random_Forest_Regression',
                                                'PhanNhomMedianIncome']) 

def ShowResult (some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test):
    st.write("Predictions:", some_data_prepared)
    st.write("Labels:", list(some_labels))
    st.write('\n')

    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train) 
    
    st.write('Sai số bình phương trung bình - cross-validation:')
    st.write("Mean: %.2f" % (rmse_cross_validation.mean()))
    st.write("Standard deviation: %.2f" % (rmse_cross_validation.std()))

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    st.write('Sai số bình phương trung bình - test:')
    st.write('%.2f' % rmse_test)

if app_mode=='Decision_Tree_Regression':
    st.title("Decision Tree Regression") 
    from pages.CaliHousing1.Decision_Tree_Regression import *

    some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test = Decision_Tree_Regression()
    ShowResult(some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test)

elif app_mode == 'Linear_Regression':
    st.title('Linear Regression')
    from pages.CaliHousing1.Linear_Regression import *
     
    some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test = Linear_Regression()
    ShowResult(some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test)

elif app_mode == 'Random_Forest_Regression_Grid_Search_CV':
    st.title('Random Forest Regression Grid Search CV')
    from pages.CaliHousing1.Random_Forest_Regression_Grid_Search_CV import *
     
    some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test = Rand_Forest_Regression_Grid_Search_CV()
    ShowResult(some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test)

elif app_mode == 'Random_Forest_Regression_Random_Search_CV':
    st.title('Random Forest Regression Random Search CV')
    from pages.CaliHousing1.Random_Forest_Regression_Random_Search_CV import *
    some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test = Rand_Forest_Regression_Rand_Search_CV()
    ShowResult(some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test)

elif app_mode == 'Random_Forest_Regression':
    st.title('Random Forest Regression')
    from pages.CaliHousing1.Random_Forest_Regression import *
    some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test = Rand_Forest_Regression_Rand_Search_CV()
    ShowResult(some_labels, some_data_prepared, rmse_train, rmse_cross_validation, rmse_test)

elif app_mode == 'PhanNhomMedianIncome':    
    st.title('Phân nhóm Median Income')
    from pages.CaliHousing1.PhanNhomMedianIncome import *
    fig, ax = plt.subplots()
    housing["income_cat"].hist()
    ax.hist(housing["income_cat"])
    st.pyplot(fig)
    st.write(housing)