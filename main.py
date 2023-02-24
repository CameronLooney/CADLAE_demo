import streamlit as st
import pandas as pd
# Set page title
st.set_page_config(page_title='CADLAE', page_icon=':robot_face:')

# Set page header
st.header('CADLAE Framework Demo')

df = pd.read_csv('/Users/cameronlooney/PycharmProjects/CADLAE_streamlit/test_data.csv')
# Add some text to the page
st.write('This is a basic example of a Streamlit web page.')

# Run the app
