# openclassrooms-P5-API_categoriser-automatiquement-des-questions
Le projet contenu dans ce repository a été réalisé dans le cadre de la formation d'ingénieur Machine learning. <br>
L'objectif de ce projet est de développer une API de prédiction de tags à partir des questions soumis par les utilisateurs du site Stackoverflow.
Le contenu du repository est le suivant:
- un fichier requirements.txt qui contient les dépendances necessaires au deploiement de l'api
- un fichier utils.py qui contient nos fonctions de pré traitements de texte et qui sera importé dans notre fichier principale streamlit_app.py
- le fichier streamlit_app.py qui correspond à l'interface ou l'entrée de notre api qui permet de réaliser les prédictions de tags
- notre modele selectionné tag_predictor_bow_model.pkl qui correspond a notre modèle de type Bow (qui utilise une régression logistique avec la stratégie OvR)
- le fichier mlb_bow_model.pkl qui nous permet de convertir les vecteurs de tags prédits en une chaine compréhensible.
- Deux scripts qui nous permis d'installer les ressources spacy et nltk dans l'environnement streamlite cloud : download_spacy_model.py et download_nltk_resources.py

Un second repository contient les scripts de chargement des questions de stackoverflow, d'exploration et de modelisation.