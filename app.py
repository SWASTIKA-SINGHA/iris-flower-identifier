import streamlit as st
from project_py import predict_iris

st.set_page_config(page_title="Iris Flower Identifier 🌸", layout="centered")

st.title("🌸 Iris Flower Identifier")
st.write("Enter the flower's measurements below to predict its species:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

if st.button("Predict"):
    result = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"🌼 The predicted Iris species is: **{result}**")
