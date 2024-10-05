import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Mall customers data set")
st.text("welcome to your model")

# Load the trained unsupervised model

model = pickle.load(open('km.pkl', 'rb'))


Gender = st.number_input(label = "Gender")
Age = st.number_input(label = "Age")
Annual_Income = st.number_input(label = "Annual Income (k$)")
Spending_Score = st.number_input(label = "Spending Score")

button = st.button("predict")

select_model = st.radio("Select a Clustering Model", ("K-Means", "Hierarchical", "DBSCAN"))

if button:
    if select_model == "K-Means":
        model = pickle.load(open('km.pkl', 'rb'))
        result = model.predict([[Gender, Age, Annual_Income, Spending_Score]])
        st.text("prediction")
        st.text(result)

    elif select_model == "DBSCAN":
        model = pickle.load(open('dbscan.pkl', 'rb'))
        result = model.fit_predict([[Gender, Age, Annual_Income, Spending_Score]])
        st.text("prediction")
        st.text(result)

    else:
        model = pickle.load(open('agg.pkl', 'rb'))
#        result = model.fit_predict([petal_length, petal_width, sepal_length, sepal_width])
        result = model.labels_
        st.text("prediction")
        st.text(result)