# Importations nécessaires
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import List
from utils import preprocess_text
import uvicorn
import numpy as np
import pandas as pd

# Créer une instance de l'application FastAPI
app = FastAPI(title='API de prédiction de tags des posts Stack Overflow',
              description='Renvoie les tags liés au post',)

@app.get("/")
def root():
    return {"message": "Welcome to the API. Check /docs for usage"}

# Chargement des modèles pré-entraînés
bow_model = joblib.load('tag_predictor_bow_model.pkl')
mlb_job = joblib.load('mlb_bow_model.pkl')

# Définition de la classe Pydantic pour la requête JSON attendue
class Question(BaseModel):
    text: str

# Définition de la classe Pydantic pour la réponse JSON
class Prediction(BaseModel):
    tags: List[str]

# Définition de la route de l'API pour la prédiction des tags
@app.post("/predict", response_model=Prediction)
async def predict_tags(question: Question):
    text = question.text
    print("hello aaaaaazizzzzzzsD")
    
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
    
    return {"tags": predicted_tags_list}