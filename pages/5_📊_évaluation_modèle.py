import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score


def fetch_data():
    """
    Fonction pour récupérer les données depuis l'état de session.
    Retourne le dataframe stocké dans st.session_state['dataframe'] ou affiche un avertissement s'il n'est pas trouvé.
    """
    if 'dataframe' in st.session_state:
        return st.session_state['dataframe']
    else:
        st.warning("Aucun dataframe trouvé dans l'état de session.")
        return None


def main():
    """
    Fonction principale pour l'évaluation des modèles.
    Permet de sélectionner des caractéristiques et une colonne cible, d'entraîner plusieurs modèles,
    d'évaluer leur performance et d'afficher le meilleur modèle avec son score.
    """
    st.title("Évaluation des Modèles")

    # Récupération du dataframe depuis l'état de session
    df = fetch_data()

    if df is not None:
        # Sélection des caractéristiques (Multiselect)
        st.write("Sélectionner les Caractéristiques:")
        selected_features = st.multiselect("Sélectionner les caractéristiques à inclure dans le modèle", df.columns.tolist())

        # Sélection de la cible (Dropdown)
        st.write("Sélectionner la Colonne Cible:")
        target_column = st.selectbox("Sélectionner la colonne cible", df.columns.tolist())

        st.header("Résultats de l'Évaluation des Modèles")

        # Initialisation des modèles
        models = {
            "Régression Linéaire": LinearRegression(),
            "Régression Logistique": LogisticRegression(max_iter=1000),
            "Arbre de Décision": DecisionTreeClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "Forêt Aléatoire": RandomForestClassifier()
        }

        # Évaluation de chaque modèle
        best_model = None
        best_score = -1
        best_model_name = ""

        for model_name, model in models.items():
            st.subheader(model_name)

            # Séparation des données en caractéristiques et cible
            X = df[selected_features]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Entraînement du modèle et évaluation
            if isinstance(model, LinearRegression):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                st.write(f"Score R²: {score:.2f}")
                st.write("""
                **Interprétation**: Le score R² mesure à quel point le modèle de régression linéaire s'adapte aux données.
                Une valeur de R² plus élevée (proche de 1.0) indique que le modèle explique une plus grande proportion de la variance de la variable cible.
                """)
            elif isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement prédites parmi toutes les instances.
                Elle indique à quel point le modèle peut classer ou prédire les résultats en fonction des caractéristiques d'entrée.
                """)
            elif isinstance(model, SVC):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement classées parmi toutes les instances.
                Elle évalue la capacité du modèle à distinguer entre différentes classes en utilisant la fonction de noyau choisie et le paramètre de régularisation.
                """)
            elif isinstance(model, GaussianNB):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement classées parmi toutes les instances.
                Naive Bayes suppose que les caractéristiques sont conditionnellement indépendantes, ce qui le rend adapté à la classification de texte ou lorsque les caractéristiques sont relativement indépendantes.
                """)

            # Suivi du meilleur modèle en fonction du score
            if isinstance(model, (LinearRegression, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)):
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name
            elif isinstance(model, SVC) or isinstance(model, GaussianNB):
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_model_name = model_name

        # Affichage du meilleur modèle et de sa performance
        if best_model:
            st.header("Meilleur Modèle")
            st.write(f"Le meilleur modèle est: {best_model_name}")
            st.write(f"Avec un score de: {best_score:.2f}")
        else:
            st.warning("Aucun modèle évalué. Veuillez vérifier vos sélections.")


if __name__ == "__main__":
    main()
