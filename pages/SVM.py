import streamlit as st
st.markdown("# SVM ❄️")
st.sidebar.markdown("# SVM ❄️")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['SVM1','SVM2']) 

if app_mode=='SVM1':
    st.title("SVM") 
    
elif app_mode == 'SVM2':
    st.title('SVM2')
    
