import streamlit as st
import spacy
import numpy
import pandas as pd
#from utils import preprocess_text  # Assurez-vous que utils.py est dans le même répertoire

st.title("Test Spacy Import")

# Afficher les versions des bibliothèques
st.write(f"Version de Spacy: {spacy.__version__}")
st.write(f"Version de Numpy: {numpy.__version__}")
st.write(f"Version de Pandas: {pd.__version__}")

# Charger le modèle spacy
try:
    nlp = spacy.load("en_core_web_sm")
    st.write("Modèle Spacy chargé avec succès.")
except OSError:
    st.write("Le modèle Spacy 'en_core_web_sm' n'est pas installé. Installation en cours...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    st.write("Modèle Spacy chargé avec succès après installation.")


# Charger les modèles pré-entraînés
try:
    bow_model = joblib.load('tag_predictor_bow_model.pkl')
    st.write("Modèle BoW chargé avec succès.")
except Exception as e:
    st.write(f"Erreur lors du chargement du modèle BoW: {e}")

try:
    mlb_job = joblib.load('mlb_bow_model.pkl')
    st.write("Modèle MultiLabelBinarizer chargé avec succès.")
except Exception as e:
    st.write(f"Erreur lors du chargement du modèle MultiLabelBinarizer: {e}")


# Analyser un texte d'exemple
try:
    doc = nlp("Hello, world!")
    st.write("Texte analysé avec succès.")
    # Afficher le texte analysé
    st.write(doc.text)
except Exception as e:
    st.write(f"Erreur lors de l'analyse du texte: {e}")
