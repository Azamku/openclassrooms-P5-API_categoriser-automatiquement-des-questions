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

# Chargement des modèles pré-entraînés
bow_model = joblib.load('tag_predictor_bow_model.pkl')
mlb_job = joblib.load('mlb_bow_model.pkl')
nlp = spacy.load('en_core_web_sm')

if st.button("Predict"):

	#response = requests.post("http://localhost:8000/predict/", data={"text": text_input})

	# Prétraiter le texte et le joindre en une chaîne de caractères
    text_cleaned_list = preprocess_text(text)
    print("text_cleaned_list: ", text_cleaned_list)
    
    # Joindre les mots prétraités en une seule chaîne de caractères
    text_cleaned = ' '.join(text_cleaned_list)
    print("text_cleaned: ", text_cleaned)
    
    # Le modèle s'attend à recevoir une liste ou un tableau de textes
    vectorizer = bow_model.named_steps['vectorizer']
    print("vectorizer vocabulary: ", vectorizer.vocabulary_)
    
    # Vérifier quels termes de text_cleaned sont dans le vocabulaire
    terms_in_vocab = [term for term in text_cleaned_list if term in vectorizer.vocabulary_]
    print("terms_in_vocab: ", terms_in_vocab)
    
    text_vectorized = vectorizer.transform([text_cleaned])  
    text_vectorized_array = text_vectorized.toarray()
    print("texte vectorisé : ", text_vectorized_array)  # Afficher le tableau pour déboguer
    
    # Créer un DataFrame pour inspecter les termes activés
    df_vectorized = pd.DataFrame(text_vectorized_array, columns=vectorizer.get_feature_names_out())
    non_zero_columns = df_vectorized.loc[:, (df_vectorized != 0).any(axis=0)]
    print("Non-zero columns: \n", non_zero_columns)
    
    # Prédiction avec le modèle de classification
    classifier = bow_model.named_steps['classifier']
    predicted_tags = classifier.predict(text_vectorized)
    print("predicted_tags: ", predicted_tags)  # Afficher les prédictions brutes pour déboguer
    
    # Inverse transform des prédictions
    predicted_tags_inverse = mlb_job.inverse_transform(predicted_tags)
    print("predicted_tags_inverse: ", predicted_tags_inverse)  # Afficher les prédictions inverses pour déboguer
    
    # Conversion des tags prédits en liste de chaînes de caractères
    predicted_tags_list = [tag for tags in predicted_tags_inverse for tag in tags]
    
    st.write(predicted_tags_list)
else:

		st.write("Erreur dans la prediction")