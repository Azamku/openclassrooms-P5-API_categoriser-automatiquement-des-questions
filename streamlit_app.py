import streamlit as st
import spacy
import numpy
import pandas as pd

st.title("Test Spacy Import")

# Afficher les versions des bibliothèques
st.write(f"Version de Spacy: {spacy.__version__}")
st.write(f"Version de Numpy: {numpy.__version__}")
st.write(f"Version de Pandas: {pd.__version__}")

# Charger le modèle spacy
try:
    nlp = spacy.load("en_core_web_sm")
    st.write("Modèle Spacy chargé avec succès.")
except Exception as e:
    st.write(f"Erreur lors du chargement du modèle Spacy: {e}")

