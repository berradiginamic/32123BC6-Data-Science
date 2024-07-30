import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np
import uuid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def fetch_data():
    """
    Fonction pour récupérer les données depuis l'état de session.
    Retourne le dataframe stocké dans st.session_state['final_dataframe'] ou affiche un avertissement s'il n'est pas trouvé.
    """
    if 'final_dataframe' in st.session_state:
        return st.session_state['final_dataframe']
    else:
        st.warning("Aucun dataframe trouvé dans l'état de session.")
        return None


def preprocess_data(df, selected_features, target_column):
    """
    Prétraitement des données : gestion des valeurs manquantes et transformation des colonnes catégorielles.
    """
    # Afficher les colonnes du DataFrame pour diagnostic
    st.write("Colonnes du DataFrame:", df.columns.tolist())

    # Gestion des valeurs manquantes
    df = df[selected_features + [target_column]].copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encodage des variables catégorielles
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

    # Afficher les colonnes après encodage
    st.write("Colonnes après encodage:", df_encoded.columns.tolist())

    # Mise à jour des features sélectionnés après encodage
    selected_features_encoded = [col for col in selected_features if col in df_encoded.columns]

    # Vérifier les colonnes après encodage
    missing_features = [col for col in selected_features if col not in df_encoded.columns]
    if missing_features:
        st.warning(f"Les colonnes sélectionnées après encodage sont manquantes : {missing_features}")

    # Séparation des caractéristiques et de la cible
    X = df_encoded[selected_features_encoded]
    y = df_encoded[target_column]

    return X, y



def main():
    """
    Fonction principale pour l'entraînement des modèles de machine learning.
    Permet de sélectionner des caractéristiques et une colonne cible, de choisir un modèle,
    d'entraîner le modèle sélectionné et de télécharger le modèle entraîné.
    """
    # Récupérer les modèles entraînés existants depuis l'état de session, ou initialiser s'ils ne sont pas présents
    trained_models = st.session_state.get('trained_models', {})

    st.title("Entraînement des Modèles de Machine Learning")

    st.header("Sélection des Caractéristiques et de la Cible")

    # Récupérer le dataframe depuis l'état de session
    df = fetch_data()

    if df is not None:
        # Sélection des caractéristiques (Multiselect)
        st.write("Sélectionner les Caractéristiques:")
        selected_features = st.multiselect("Sélectionner les caractéristiques à inclure dans le modèle", df.columns.tolist())

        # Sélection de la cible (Dropdown)
        st.write("Sélectionner la Colonne Cible:")
        target_column = st.selectbox("Sélectionner la colonne cible", df.columns.tolist())

        st.header("Sélection du Modèle")

        # Suggérer des modèles en fonction des caractéristiques du jeu de données
        numeric_cols = df.select_dtypes(include=['number']).shape[1]
        text_cols = df.select_dtypes(include=['object']).shape[1]
        categorical_data = False  # Placeholder pour la vérification des données catégorielles (à implémenter en fonction de vos données)
        data_size = df.shape[0]

        if numeric_cols > 0:
            st.write("Colonnes Numériques Détectées.")
            st.write("Modèles Suggérés:")
            st.write("- Régression Linéaire")
            st.write("- Arbre de Décision")
            st.write("- SVM")

        if text_cols > 0:
            st.write("Colonnes Textuelles Détectées.")
            st.write("Modèles Suggérés:")
            st.write("- Naive Bayes")

        if categorical_data:
            st.write("Données Catégorielles Détectées.")
            st.write("Modèles Suggérés:")
            st.write("- Régression Logistique")
            st.write("- Arbre de Décision")
            st.write("- Forêt Aléatoire")

        st.header("Entraînement du Modèle")

        model_name = st.selectbox("Sélectionner le Modèle", ["Régression Linéaire", "Régression Logistique",
                                                            "Arbre de Décision", "SVM", "Naive Bayes",
                                                            "Forêt Aléatoire"])

        if model_name == "Régression Linéaire":
            st.write("Ajuster les Paramètres:")
            fit_intercept = st.checkbox("Ajuster l'intercept", value=True)
            model = LinearRegression(fit_intercept=fit_intercept)

        elif model_name == "Régression Logistique":
            st.write("Ajuster les Paramètres:")
            C = st.slider("Force de régularisation (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            model = LogisticRegression(C=C, max_iter=1000)

        elif model_name == "Arbre de Décision":
            st.write("Ajuster les Paramètres:")
            criterion = st.selectbox("Critère", ["gini", "entropy"])
            max_depth = st.slider("Profondeur Maximale", min_value=1, max_value=20, value=3, step=1)
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

        elif model_name == "SVM":
            st.write("Ajuster les Paramètres:")
            C = st.slider("Paramètre de régularisation (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            kernel = st.selectbox("Noyau", ["linéaire", "poly", "rbf", "sigmoid"])
            model = SVC(C=C, kernel=kernel)

        elif model_name == "Naive Bayes":
            st.write("Ajuster les Paramètres:")
            model = GaussianNB()

        elif model_name == "Forêt Aléatoire":
            st.write("Ajuster les Paramètres:")
            n_estimators = st.slider("Nombre d'estimateurs", min_value=10, max_value=200, value=100, step=10)
            max_depth = st.slider("Profondeur Maximale", min_value=1, max_value=20, value=None, step=1)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        if st.button("Entraîner le Modèle"):
            if len(selected_features) == 0:
                st.warning("Veuillez sélectionner au moins une caractéristique.")
                return

            if not target_column:
                st.warning("Veuillez sélectionner une colonne cible.")
                return

            # Prétraiter les données
            X, y = preprocess_data(df, selected_features, target_column)

            model.fit(X, y)
            y_pred = model.predict(X)

            if isinstance(model, LinearRegression):
                score = r2_score(y, y_pred)
                st.write(f"Score R²: {score:.2f}")
                st.write("""
                **Interprétation**: Le score R² mesure à quel point le modèle de régression linéaire s'adapte aux données.
                Une valeur de R² plus élevée (proche de 1.0) indique que le modèle explique une plus grande proportion de la variance de la variable cible.
                """)
            elif isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)):
                accuracy = accuracy_score(y, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement prédites parmi toutes les instances.
                Elle indique à quel point le modèle peut classer ou prédire les résultats en fonction des caractéristiques d'entrée.
                """)
            elif isinstance(model, SVC):
                accuracy = accuracy_score(y, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement classées parmi toutes les instances.
                Elle évalue la capacité du modèle à distinguer entre différentes classes en utilisant la fonction de noyau choisie et le paramètre de régularisation.
                """)
            elif isinstance(model, GaussianNB):
                accuracy = accuracy_score(y, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement classées par le classificateur Naive Bayes.
                Ce modèle est basé sur l'hypothèse d'indépendance entre les caractéristiques et peut être particulièrement efficace avec des données de grande dimension.
                """)

            # Sauvegarde du modèle
            filename = f"{model_name}_model.pkl"
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
            st.write(f"Modèle entraîné et sauvegardé sous {filename}")

if __name__ == "__main__":
    main()
