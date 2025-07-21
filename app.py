import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load iris data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Identifier")
st.write("Enter the flower measurements to predict the Iris species:")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)

if st.button("Predict"):
    prediction = clf.predict(input_data)[0]
    species = iris.target_names[prediction]
    st.success(f"🌼 Predicted Iris Species: **{species.capitalize()}**")
