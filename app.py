import streamlit as st
from project_py import predict_species

st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

st.title("🌸 Iris Flower Species Predictor")

st.write("Enter the following details to predict the species:")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"The predicted Iris species is: **{prediction}**")
