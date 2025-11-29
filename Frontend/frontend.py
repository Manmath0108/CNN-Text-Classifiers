import streamlit as st
import requests

st.title('Text Classifier with TextCNN')

text_input = st.text_area('Enter Text for Classification')

if st.button('Classify Text'):
    if text_input:

        api_url = 'http://textcnn-classifier-v1-0.onrender.com'

        response = requests.post(api_url, json={'text': text_input})
        
        if response.status_code == 200:
            result = response.json()
            st.subheader('Classification Result:')
            st.write(result)
        else:
            st.error('Error: Unable to get classification from the API')
    else:
        st.warning('Please enter some text to classify.')
