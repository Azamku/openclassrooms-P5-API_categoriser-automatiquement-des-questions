import streamlit as st
import requests
import joblib
import utils
import spacy
import subprocess
import os

# Telecharger le modèle SpaCy si nécessaire
def install_spacy_model():
    model_path = os.path.join(spacy.util.get_package_path("spacy"), "data", "en_core_web_sm")
    if not os.path.exists(model_path):
        subprocess.run(["python", "download_spacy_model.py"])

install_spacy_model()


st.title("API FastAPI avec Streamlit !!!!!")
text_input=st.text_input("Entrer le texte pour la prediction de tags")
