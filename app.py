import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import
RandomForestClassifier
st.title("iris flower type identifier")
iris=load_iris()
X=pd.DataFrame(iris.data,columns=iris.feature_names)
y=iris.target
model=RandomForestClassifier()
model.fit(X,y)
st.sidebar.head("Enter flower features")
sepal_length= st.sidebar.slider('sepal length (cm)',4.0,8.0,5.0)
sepal_width=st.sidebar.slider('sepal width (cm)',2.0,4.5,3.0)
petal_length=st.sidebar.slider('petal length (cm)',1.0,7.0,4.0)
petal_width=st.sidebar.slider('petal width (cm)',0.1,2.5,1.0)
features=[[sepal_length,sepal_width,petal_length,petal_width]]
prediction=model.predict(features)
predicted_class=iris.target_names[predictions[0]]
st.write(f"###predicted iris type:**{predicted_class}**")
