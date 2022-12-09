import streamlit as st
st.markdown("# KNN ❄️")
st.sidebar.markdown("# KNN ❄️")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['KNN1','KNN2']) 

if app_mode=='KNN1':
    st.title("KNN1") 
    

elif app_mode == 'KNN2':
    st.title('KNN02')
    
