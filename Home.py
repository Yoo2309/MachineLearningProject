import streamlit as st

st.markdown("# ĐỒ ÁN CUỐI KỲ HỌC MÁY 🎈")
st.sidebar.markdown("# Home 🎈")    
container = st.container(); 

st.write('Lớp: MALE431984_22_1_04')
st.write('SINH VIÊN THỰC HIỆN:')
col1,col2 = st.columns([15,20])
with col1:
    text = '''
    Họ và tên
    Trần Ngô Bích Du
    Đào Thị Thanh Vi'''
    st.text(text)
with col2:
    code = '''
    Mã số sinh viên
    20110618
    Đào Thị Thanh Vi'''
    st.text(code)
st.write('GIÁO VIÊN HƯỚNG DẪN: Trần Tiến Đức')

st.image('images/machineLearning3.png')