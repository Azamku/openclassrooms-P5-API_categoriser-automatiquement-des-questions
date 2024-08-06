import streamlit as st
import requests
import joblib
import utils
import spacy
import subprocess
import os

# Telecharger le modèle SpaCy si nécessaire
def install_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "download_spacy_model.py"])
        spacy.load("en_core_web_sm")

install_spacy_model()


st.title("API FastAPI avec Streamlit !!!!!")
text_input=st.text_input("Entrer le texte pour la prediction de tags")
