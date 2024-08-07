# pretraitement version spacy :
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime
#from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
import nltk
import os
import pandas as pd
import re
#import spacy
import sys
#import torch


# Telecharger les stopwords et tokenizer de NLTK
# nltk.download('stopwords')
# nltk.download('punkt')

try:
    from bs4 import BeautifulSoup
    print("BeautifulSoup importé avec succès.")
except ImportError as e:
    print(f"Erreur lors de l'importation de BeautifulSoup: {e}")

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    print("NLTK et ressources importés avec succès.")
except ImportError as e:
    print(f"Erreur lors de l'importation de NLTK: {e}")


# Fonction pour nettoyer le texte HTML et enlever les portions de code
def clean_html_code(text):
    # Supprimer les portions de code
    # Supprimer les balises <code> et leur contenu
    text = re.sub(r'<code>.*?</code>', '', text, flags=re.DOTALL)
    # Supprimer les balises <p> en conservant le contenu
    text = re.sub(r'</?p>', '', text)
    text = re.sub(r'\n', ' ', text)
    # Utiliser BeautifulSoup pour nettoyer les balises HTML
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

# Fonction pour normaliser le texte
def normalize_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer la ponctuation et les caract�res sp�ciaux
    text = re.sub(r'\W+', ' ', text)
    # Supprimer plusieurs espaces par un espace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Fonction de pr�traitement complet du texte
def preprocess_text(text,nlp):
# Charger le mod�le anglais de SpaCy
    #nlp = spacy.load('en_core_web_sm')

# Initialiser le stemmer de NLTK
    stemmer = PorterStemmer()

# Ajouter des stopwords personnalis�s
    custom_stopwords = set([
    'like', 'question', 'use', 'want', 'one', 'know', 'work', 'example', 'code', 'seem', 
    'using', 'instead', 'way', 'get', 'would', 'need', 'following', '1', '2', 'run', 
    'something', 'trying', 'tried', 'also', 'new', 'could', 'see', 'line', 'however', 
    'solution', '3', '4', '5', 'without', 'still', 'answer', 'say', 'another', 'help', 
    'anyone', 'best', 'looking', 'show', 'give', 'better', 'many', 'good', 'even', 
    'think', 'thing', 'look', 'problem', 'try', 'possible'
    ])
    nltk_stopwords = set(stopwords.words('english'))
    all_stopwords = nltk_stopwords.union(custom_stopwords)
    # Nettoyage du texte HTML et suppression des portions de code
    #text = clean_html_code(text)
    # Normalisation
    #text = normalize_text(text)
    # Utilisation de SpaCy pour lemmatisation et POS tagging
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and token.text not in all_stopwords]
    # Stemming des tokens (si n�cessaire)
    #tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Fonction de pr�traitement complet du texte (pour use,bert,w2vec)
def preprocess_text_NN(text):
    # Nettoyage du texte HTML et suppression des portions de code
    text = clean_html_code(text)
    # Normalisation
    text = normalize_text(text)
    tokens=text.split()
    return tokens
