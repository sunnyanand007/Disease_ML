import streamlit as st 
import pandas as pd 
from sklearn.svm import SVC


st.title("Disease Detection Using SVM")


data = pd.read_csv("temp.csv")

st.write(data)


x = data[['Temperature','Heart_Rate']]
y = data['Disease'].map({'Yes':1,'No':0})



model = SVC()
model.fit(x,y)


temperature = st.number_input("Enter the temperature:",97.0,104.0,step=1.0)
heart_rate = st.number_input("Enter the heart-rate:",70.0,95.0,step=1.0)


prediction = model.predict([[temperature,heart_rate]])[0]

result = 'Yes' if prediction == 1 else 'No'

st.write(result)


