import streamlit as st
import spacy
import numpy
import pandas as pd
from utils import preprocess_text  # Assurez-vous que utils.py est dans le même répertoire
import joblib

import subprocess
import sys

import platform
# Afficher la version de Python
python_version = platform.python_version()
st.write(f"La version de Python utilisée est : {python_version}")


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
        st.write("text_cleaned_joined: ", text_cleaned_joined)

        # debut debogue
        # Le modèle s'attend à recevoir une liste ou un tableau de textes
        # vectorizer = bow_model.named_steps['vectorizer']
        # #st.write("vectorizer vocabulary: ", vectorizer.vocabulary_)
    
        # # Vérifier quels termes de text_cleaned sont dans le vocabulaire
        # terms_in_vocab = [term for term in text_cleaned_list if term in vectorizer.vocabulary_]
        # st.write("terms_in_vocab: ", terms_in_vocab)

        # text_vectorized = vectorizer.transform(text_cleaned_list)  
        # text_vectorized_array = text_vectorized.toarray()
        # #st.write("texte vectorisé : ", text_vectorized_array)  # Afficher le tableau pour déboguer

        # # Créer un DataFrame pour inspecter les termes activés
        # df_vectorized = pd.DataFrame(text_vectorized_array, columns=vectorizer.get_feature_names_out())
        # non_zero_columns = df_vectorized.loc[:, (df_vectorized != 0).any(axis=0)]
        # st.write("Non-zero columns: \n", non_zero_columns)

        # # Prédiction avec le modèle de classification
        # classifier = bow_model.named_steps['classifier']
        # predicted_tags = classifier.predict(text_vectorized)
        # st.write("predicted_tags: ", predicted_tags)  # Afficher les prédictions brutes pour déboguer
    
        # # Inverse transform des prédictions
        # predicted_tags_inverse = mlb_job.inverse_transform(predicted_tags)
        # st.write("predicted_tags_inverse: ", predicted_tags_inverse)  # Afficher les prédictions inverses pour déboguer
    
        # # Conversion des tags prédits en liste de chaînes de caractères
        # predicted_tags_list = [tag for tags in predicted_tags_inverse for tag in tags]    
        # st.write("predicted_tags_list: ",predicted_tags_list)
        #fin debogue

        bow_predict_result=bow_model.predict([text_cleaned_joined])
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
