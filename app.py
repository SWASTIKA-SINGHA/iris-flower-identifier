import streamlit as st
from project_py import predict_iris

st.set_page_config(page_title="Iris Flower Identifier")

st.title("🌸 Iris Flower Identifier")
st.write("Enter the measurements below to predict the type of Iris flower:")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    result = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"🌼 Predicted Iris Flower Type: **{result}**")
