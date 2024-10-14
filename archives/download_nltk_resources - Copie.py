import nltk
import os

# Spécifier le répertoire de téléchargement
nltk_data_dir = './nltk_data'
nltk.data.path.append(nltk_data_dir)

# Fonction pour vérifier l'existence des fichiers NLTK téléchargés
def check_nltk_resource(resource_name, resource_path):
    if os.path.exists(resource_path):
        print(f"{resource_name} téléchargé avec succès.")
    else:
        print(f"Erreur: {resource_name} n'a pas été téléchargé.")

# Télécharger les stopwords et le tokenizer de NLTK
print("Téléchargement des stopwords...")
nltk.download('stopwords', download_dir=nltk_data_dir)
check_nltk_resource('Stopwords', os.path.join(nltk_data_dir, 'corpora', 'stopwords'))

print("Téléchargement de punkt...")
nltk.download('punkt', download_dir=nltk_data_dir)
check_nltk_resource('Punkt', os.path.join(nltk_data_dir, 'tokenizers', 'punkt'))

print("Téléchargement de wordnet...")
nltk.download('wordnet', download_dir=nltk_data_dir)
check_nltk_resource('WordNet', os.path.join(nltk_data_dir, 'corpora', 'wordnet'))

print("Tous les téléchargements sont terminés.")
