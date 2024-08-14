import streamlit as st
import spacy
import numpy
import pandas as pd
from utils import preprocess_text  # Assurez-vous que utils.py est dans le même répertoire
import joblib

import subprocess
import sys

st.write(f"debut code telech nltk: ")
# Assurez-vous que les ressources NLTK sont téléchargées
try:
    result = subprocess.run(
        [sys.executable, "download_nltk_resources.py"],
        check=True,
        capture_output=True,
        text=True
    )
    print("NLTK resources downloaded successfully.")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error while downloading NLTK resources: {e}")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
st.write(f"fin code telech nltk")

# Afficher les versions des bibliothèques
st.write(f"Version de Spacy: {spacy.__version__}")
st.write(f"Version de Numpy: {numpy.__version__}")
st.write(f"Version de Pandas: {pd.__version__}")


# Vérifier l'importation de preprocess_text
try:
    from utils import preprocess_text
    st.write("Fonction preprocess_text importée avec succès.")
except ImportError as e:
    st.write(f"Erreur lors de l'importation de preprocess_text: {e}")
    

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

# Importer joblib avec gestion des erreurs pour débogage
try:
    import joblib
    st.write("joblib importé avec succès.")
except ModuleNotFoundError as e:
    st.write(f"Erreur lors de l'importation de joblib: {e}")

# Importer scikit-learn avec gestion des erreurs pour débogage
try:
    import sklearn
    st.write("scikit-learn importé avec succès.")
except ModuleNotFoundError as e:
    st.write(f"Erreur lors de l'importation de scikit-learn: {e}")

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

st.title("API FastAPI avec Streamlit !!!!!")
text_input = st.text_input("Entrer le texte pour la prédiction de tags")

if st.button("Predict"):
    if text_input:
        # Prétraiter le texte et le joindre en une chaîne de caractères
        text_cleaned_list = preprocess_text(text_input)
        st.write("text_cleaned_list: ", text_cleaned_list)
        # Joindre les mots prétraités en une seule chaîne de caractères
        text_cleaned_joined = ' '.join(text_cleaned_list)
        print("text_cleaned_joined: ", text_cleaned_list)
        bow_predict_result=bow_model.predict(text_cleaned_list)
        st.write("tags predits:",bow_predict_result)
        tags_predits=mlb_job.inverse_transform(bow_predict_result)
        st.write("tags_predits apres inverse:",tags_predits)
    else:
        st.write("Veuillez entrer du texte pour la prédiction.")
else:
    st.write("Cliquez sur le bouton pour obtenir une prédiction.")


# Analyser un texte d'exemple
# try:
#     doc = nlp("Hello, world!")
#     st.write("Texte analysé avec succès.")
#     # Afficher le texte analysé
#     st.write(doc.text)
# except Exception as e:
#     st.write(f"Erreur lors de l'analyse du texte: {e}")
