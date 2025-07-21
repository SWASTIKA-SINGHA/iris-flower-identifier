import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def main():
    st.title("🌸 Iris Flower Classification App")

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    clf = RandomForestClassifier()
    clf.fit(X, y)

    st.write("Enter flower measurements to predict its species:")

    sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.0)
    petal_length = st.slider("Petal length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal width", 0.1, 2.5, 1.0)

    if st.button("Predict"):
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = clf.predict(input_data)
        predicted_species = iris.target_names[prediction[0]]
        st.success(f"The predicted Iris species is: **{predicted_species}**")

if __name__ == '__main__':
    main()
