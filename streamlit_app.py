import streamlit as st
import requests

st.title("API FastAPI avec Streamlit")

text_input = st.text_input("Entrez le texte pour prédiction")

if st.button("Predict"):
    if text_input:
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": text_input})
        if response.status_code == 200:
            st.write(response.json())
        else:
            st.write("Erreur dans la prédiction")
    else:
        st.write("Veuillez entrer un texte.")
