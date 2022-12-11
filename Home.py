import streamlit as st

st.sidebar.markdown("# 🏠 Home")
container = st.container(); 

html =  '''
           <h2 style=\"font-weight: bold; text-align: center;\">ĐỒ ÁN CUỐI KỲ MACHINE LEARNING</h2>
           <h4 style=\"font-weight: bold;\">Mã lớp học phần: MALE431984_22_1_04</h4>
           <h4 style=\"font-weight: bold;\">Giảng viên hướng dẫn: Ths.Trần Tiến Đức</h4>
           <h4 style=\"font-weight: bold;\">Sinh viên thực hiện:</h4>
           <h4 style=\"font-weight: bold; text-align: center;\">Trần Ngô Bích Du &nbsp;&nbsp;&nbsp;&nbsp; 20110618</h4>
           <h4 style=\"font-weight: bold; text-align: center;\">Đào Thị Thanh Vi &nbsp;&nbsp;&nbsp;&nbsp; 20110223</h4>
        '''
st.markdown(html, unsafe_allow_html=True)

st.image('images/machineLearning3.png')