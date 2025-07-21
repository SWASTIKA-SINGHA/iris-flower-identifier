import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Flower Classification")

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

sepal_length = st.slider('Sepal length', 4.0, 8.0)
sepal_width = st.slider('Sepal width', 2.0, 4.5)
petal_length = st.slider('Petal length', 1.0, 7.0)
petal_width = st.slider('Petal width', 0.1, 2.5)

if st.button("predict"):
  prediction=clf.predict([[sepal_length,sepal_width,petal_length,petal_width]])
  st.success(f"predicted iris type:{iris.target_names[prediction[0]]}")

