import streamlit as st
from project_py import predict_iris

st.title("Iris Flower Identifier 🌸")

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, step=0.1)

if st.button("Predict"):
    prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"The predicted Iris flower is: **{prediction}**")
