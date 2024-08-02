import streamlit as st
import requests

st.title("API FastAPI avec Streamlit !!!!!")
text_input=st.text_input("Entrer le texte pour la prediction de tags")
if st.button("Predict"):

	response = requests.post("http://localhost:8000/predict", json={"text": text_input})

	if response.status_code == 200:

		st.write(response.json())
	else:

		st.write("Erreur dans la prediction")