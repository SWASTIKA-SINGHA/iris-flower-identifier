import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Identifier")
st.write("Upload feature inputs below to classify the Iris flower type.")

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

# Input sliders
sepal_length = st.slider("Sepal length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(input_data)
predicted_species = iris.target_names[prediction][0]

st.success(f"The predicted Iris species is: **{predicted_species.capitalize()}**")
