import sys
# this checks the running evn path
# print(sys.executable)
# import analyze_sentiment_with_shap from predict.py
from predict import analyze_sentiment_with_shap

# print(analyze_sentiment_with_shap("I love this product!")) # use this if the fuction only returns the sentiment not if the function returns the shap plot adjust the function to see it print


from streamlit_shap import st_shap
import streamlit as st

st.title("Sentiment Analysis")

text = st.text_area("Enter a sentence:")
if st.button("Analyze"):
    sentiment, shap_plot = analyze_sentiment_with_shap(text)
    st.write("**Predicted Sentiment:**", sentiment)
    st_shap(shap_plot)

# this is the sample code to this function with streamlit (pls test this and make changes as needed)
# update the requirements.txt if you install anything
# .\amazon-env\Scripts\activate
