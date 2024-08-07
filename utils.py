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
