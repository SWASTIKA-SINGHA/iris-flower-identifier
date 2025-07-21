import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data and model
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Streamlit UI
st.set_page_config(page_title="Iris Flower Classifier")
st.title("🌸 Iris Flower Identifier")

st.write("Adjust the sliders to input flower measurements:")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_class = iris.target_names[prediction][0]
    st.success(f"🌼 The predicted Iris species is: **{predicted_class}**")
