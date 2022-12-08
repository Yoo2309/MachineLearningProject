import streamlit as st
from pages.CheckSpamToxicComment.CheckSpamToxicComment import *
st.markdown("# Spam and Toxic Comment ❄️")
st.sidebar.markdown("# Spam and Toxic Comment ❄️")


with st.form (key='predict', clear_on_submit=True):
    code = st.text_area('Enter your Comment')
    submit = st.form_submit_button("Submit")
    
if submit:
    st.info("Prediction result:")
    st.write(code + " is " + CheckComment(code))

