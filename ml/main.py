import pandas as pd
import re
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pg8000.native

# Création d'un dataset étendu pour améliorer les performances du modèle
data = {
    "text": [
        # Dataset original
        "Je te déteste, tu es horrible!",  # Haineux
        "J'aime beaucoup cette vidéo, merci.",  # Non haineux
        "Va te faire voir, imbécile.",  # Haineux
        "Quel contenu inspirant, bravo à l'équipe!",  # Non haineux
        "Tu es vraiment nul et inutile.",  # Haineux
        "Je suis impressionné par la qualité de cette vidéo.",  # Non haineux
        "Ferme-la, personne ne veut entendre ça.",  # Haineux
        "C'est une discussion constructive, merci pour vos efforts.",  # Non haineux
        "Ce commentaire est complètement stupide et inutile.",  # Haineux
        "Merci pour cette vidéo, elle m'a beaucoup aidé!",  # Non haineux
        "Personne n'a besoin de voir des bêtises pareilles.",  # Haineux
        "Excellent contenu, continuez comme ça!",  # Non haineux
        "Tu ne comprends rien, arrête de commenter.",  # Haineux
        "Bravo, c'est exactement ce que je cherchais.",  # Non haineux
        "Espèce d'idiot, tu ne sais même pas de quoi tu parles.",  # Haineux
        "Cette vidéo est très claire, merci pour le travail.",  # Non haineux
        "Tu es une honte, personne ne veut lire ça.",  # Haineux
        "Le tutoriel est super bien expliqué, merci!",  # Non haineux
        "C'est complètement débile, arrête de poster.",  # Haineux
        "J'adore cette chaîne, toujours des vidéos intéressantes.",  # Non haineux
        "Dégage d'ici, personne ne te supporte.",  # Haineux
        "Merci pour ces conseils, c'est vraiment utile.",  # Non haineux
        "T'es vraiment le pire, tes vidéos sont nulles.",  # Haineux
        "Une très bonne vidéo, claire et précise, bravo!",  # Non haineux
        
        # Nouveaux commentaires haineux
        "Tu es vraiment stupide, arrête de parler.",  # Haineux
        "Cette chaîne est nulle, comme toi d'ailleurs.",  # Haineux
        "Tu racontes n'importe quoi, espèce d'idiot.",  # Haineux
        "J'espère que ta chaîne va couler, c'est de la merde.",  # Haineux
        "Retourne à l'école avant de faire des vidéos.",  # Haineux
        "Personne ne t'aime, arrête de poster.",  # Haineux
        "Tu ne mérites pas d'être sur cette plateforme.",  # Haineux
        "Ta voix est insupportable, ferme-la.",  # Haineux
        "Je vais signaler cette vidéo, elle est pathétique.",  # Haineux
        "Les gens comme toi devraient disparaître d'internet.",  # Haineux
        "Quelle perte de temps, vidéo à éviter absolument.",  # Haineux
        "Tu n'as aucun talent, renonce à ta chaîne.",  # Haineux
        "Je n'ai jamais vu quelque chose d'aussi mauvais.",  # Haineux
        "C'est ridicule, comme ton intelligence.",  # Haineux
        "Comment oses-tu poster une telle merde?",  # Haineux
        "Ton contenu est aussi médiocre que ta personne.",  # Haineux
        "Tu es la honte de cette communauté.",  # Haineux
        "Je déteste tout ce que tu représentes.",  # Haineux
        "Cette vidéo m'a donné envie de vomir.",  # Haineux
        "Franchement, qui regarde ces conneries?",  # Haineux
        "Tu devrais avoir honte de ce contenu minable.",  # Haineux
        "Vidéo de merde, présentateur de merde.",  # Haineux
        "Je n'ai jamais rien vu d'aussi nul.",  # Haineux
        "Arrête de polluer internet avec ta présence.",  # Haineux
        "Tu es vraiment pathétique, comme ta chaîne.",  # Haineux
        "Vraiment le pire YouTubeur que j'ai jamais vu.",  # Haineux
        
        # Nouveaux commentaires non haineux
        "Super vidéo, j'ai appris beaucoup de choses!",  # Non haineux
        "Merci pour ces explications claires et précises.",  # Non haineux
        "Ton contenu est toujours de grande qualité.",  # Non haineux
        "J'attends avec impatience ta prochaine vidéo!",  # Non haineux
        "C'est exactement ce dont j'avais besoin, merci.",  # Non haineux
        "J'adore ta façon d'expliquer les choses complexes.",  # Non haineux
        "Cette vidéo mérite plus de vues, je vais la partager.",  # Non haineux
        "Excellent travail, continue comme ça!",  # Non haineux
        "Tu as un talent incroyable pour la vulgarisation.",  # Non haineux
        "Je me suis abonné directement après cette vidéo.",  # Non haineux
        "Très instructif, merci pour ton temps.",  # Non haineux
        "Ta chaîne est une vraie mine d'or d'informations.",  # Non haineux
        "Je regarde toutes tes vidéos, elles sont géniales.",  # Non haineux
        "Contenu de qualité comme toujours, bravo!",  # Non haineux
        "Tu expliques mieux que mes professeurs!",  # Non haineux
        "Merci de partager tes connaissances avec nous.",  # Non haineux
        "Très bien réalisé, j'apprécie ton travail.",  # Non haineux
        "Cette vidéo m'a beaucoup aidé, merci!",  # Non haineux
        "Je recommande ta chaîne à tous mes amis.",  # Non haineux
        "Tes vidéos sont toujours un plaisir à regarder.",  # Non haineux
        "Le montage est super, le contenu aussi!",  # Non haineux
        "J'ai enfin compris ce sujet grâce à toi.",  # Non haineux
        "Ton énergie est contagieuse, j'adore!",  # Non haineux
        "Vidéo très instructive, merci pour le partage.",  # Non haineux
        "Je suis impressionné par la qualité de ton travail.",  # Non haineux
        "Merci pour ces conseils qui vont m'être très utiles."  # Non haineux
    ],
    "label": [
        # Labels du dataset original (24 commentaires)
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        # Labels des nouveaux commentaires haineux (26 commentaires)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Labels des nouveaux commentaires non haineux (26 commentaires)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
}

# Afficher les statistiques du dataset
print(f"Dataset chargé : {len(data['text'])} commentaires au total")
print(f"Commentaires haineux : {sum(data['label'])}")
print(f"Commentaires non haineux : {len(data['label']) - sum(data['label'])}")

# Liste des mots vides en français
french_stopwords = [
    "le", "la", "les", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "elle", "en", "et", "eux", "il", "je", "la", "le", "leur", "lui", "ma",
    "mais", "me", "même", "mes", "moi", "mon", "ni", "notre", "nous", "on",
    "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son",
    "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre",
    "vous", "c", "d", "j", "l", "à", "m", "n", "s", "t", "y", "été", "étée",
    "étées", "étés", "étant", "suis", "es", "est", "sommes", "êtes", "sont",
    "serai", "seras", "sera", "serons", "serez", "seront", "serais", "serait",
    "serions", "seriez", "seraient"
]

# Configuration de la base de données
DB_CONFIG = {
    "host": "localhost",  # Connect via port forwarding from host machine
    "database": "hatespeech_db",
    "user": "hatespeech_user",
    "password": "password123",
    "port": 5432
}

def clean_text(text):
    """Nettoie le texte en le mettant en minuscule et en supprimant les caractères spéciaux"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def connect_to_db():
    """Établit une connexion à la base de données PostgreSQL"""
    try:
        conn = pg8000.native.Connection(
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            host=DB_CONFIG["host"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"]
        )
        print("Connexion à la base de données établie avec succès.")
        return conn
    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données: {e}")
        return None

def save_model_metrics(conn, metrics, model_params):
    """Enregistre les métriques du modèle dans la base de données"""
    try:
        conn.run(
            "INSERT INTO model_metrics (accuracy, precision, recall, f1_score, model_parameters) VALUES (:accuracy, :precision, :recall, :f1, :params)",
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            params=json.dumps(model_params)
        )
        print("Métriques du modèle enregistrées dans la base de données.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des métriques: {e}")

def save_comments_data(conn, df, predictions=None, probabilities=None):
    """Enregistre les commentaires et leurs prédictions dans la base de données"""
    try:
        for i, row in df.iterrows():
            confidence = None
            prediction = None
            
            if predictions is not None:
                prediction = int(predictions[i])
                
            if probabilities is not None and len(probabilities[i]) > 1:
                # Probabilité de la classe positive (haineux)
                confidence = float(probabilities[i][1])
            
            conn.run(
                "INSERT INTO comments (text, cleaned_text, hate_label, prediction, confidence) VALUES (:text, :cleaned_text, :label, :prediction, :confidence)",
                text=row["text"],
                cleaned_text=row["text_clean"],
                label=int(row["label"]),
                prediction=prediction,
                confidence=confidence
            )
        print(f"{len(df)} commentaires enregistrés dans la base de données.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des commentaires: {e}")

def main():
    # Création du DataFrame
    df = pd.DataFrame(data)
    print("Dataset chargé avec succès.")
    
    # Prétraitement des données
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Connexion à la base de données
    conn = connect_to_db()
    if conn is None:
        print("Impossible de continuer sans connexion à la base de données.")
        return
    
    # Vectorisation
    vectorizer = CountVectorizer(stop_words=french_stopwords, max_features=100)
    X = vectorizer.fit_transform(df['text_clean'])
    y = df['label']
    print("Vectorisation terminée.")
    
    # Sauvegarde du vectoriseur
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Vectoriseur sauvegardé.")
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Entraînement du modèle
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")
    
    # Sauvegarde du modèle
    with open('hate_speech_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Modèle sauvegardé.")
    
    # Évaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    
    # Calcul des métriques
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }
    
    # Paramètres du modèle
    model_params = {
        "max_features": 100,
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "test_size": 0.25,
        "random_state": 42
    }
    
    # Enregistrement des métriques dans la base de données
    save_model_metrics(conn, metrics, model_params)
    
    # Prédiction sur tous les commentaires et enregistrement dans la base de données
    all_predictions = model.predict(X)
    all_probabilities = model.predict_proba(X)
    save_comments_data(conn, df, all_predictions, all_probabilities)
    
    # Test sur de nouveaux commentaires
    new_comments = [
        "Je ne supporte pas cette personne.",  # Haineux
        "Cette vidéo est incroyable, merci pour votre travail.",  # Non haineux
        "Arrête de dire n'importe quoi, imbécile.",  # Haineux
        "Une excellente présentation, bravo à toute l'équipe."  # Non haineux
    ]
    
    # Création d'un DataFrame pour les nouveaux commentaires
    new_df = pd.DataFrame({"text": new_comments})
    new_df['text_clean'] = new_df['text'].apply(clean_text)
    
    # Vectorisation des nouveaux commentaires
    new_comments_vectorized = vectorizer.transform(new_df['text_clean'])
    new_predictions = model.predict(new_comments_vectorized)
    new_probabilities = model.predict_proba(new_comments_vectorized)
    
    # Ajout des labels prédits au DataFrame
    new_df['label'] = new_predictions
    
    # Enregistrement des nouveaux commentaires dans la base de données
    save_comments_data(conn, new_df, new_predictions, new_probabilities)
    
    print("\nPrédictions sur les nouveaux commentaires :")
    for comment, label, proba in zip(new_comments, new_predictions, new_probabilities):
        confidence = proba[1]  # Probabilité de la classe positive (haineux)
        print(f"Commentaire : '{comment}' -> {'Haineux' if label == 1 else 'Non haineux'} (confiance: {confidence:.2f})")
    
    # Fermeture de la connexion à la base de données
    conn.close()
    print("Connexion à la base de données fermée.")

if __name__ == "__main__":
    main()