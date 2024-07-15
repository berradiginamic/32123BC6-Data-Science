import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
import uuid

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

            # Séparation des données en caractéristiques et cible
            X = df[selected_features]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if isinstance(model, LinearRegression):
                score = r2_score(y_test, y_pred)
                st.write(f"Score R²: {score:.2f}")
                st.write("""
                **Interprétation**: Le score R² mesure à quel point le modèle de régression linéaire s'adapte aux données.
                Une valeur de R² plus élevée (proche de 1.0) indique que le modèle explique une plus grande proportion de la variance de la variable cible.
                """)
            elif isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)):
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement prédites parmi toutes les instances.
                Elle indique à quel point le modèle peut classer ou prédire les résultats en fonction des caractéristiques d'entrée.
                """)
            elif isinstance(model, SVC):
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement classées parmi toutes les instances.
                Elle évalue la capacité du modèle à distinguer entre différentes classes en utilisant la fonction de noyau choisie et le paramètre de régularisation.
                """)
            elif isinstance(model, GaussianNB):
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Précision: {accuracy:.2f}")
                st.write("""
                **Interprétation**: La précision mesure le pourcentage d'instances correctement classées parmi toutes les instances.
                Naive Bayes suppose que les caractéristiques sont conditionnellement indépendantes, ce qui le rend adapté à la classification de texte ou lorsque les caractéristiques sont relativement indépendantes.
                """)

            # Générer un identifiant unique pour le modèle
            model_id = str(uuid.uuid4())

            # Stocker le modèle entraîné dans l'état de session
            trained_models[model_id] = model

            # Exporter le modèle entraîné
            st.write(f"Exporter le Modèle Entraîné (ID: {model_id}):")
            filename = f"modele_entraine_{model_name}_{model_id}.pkl"
            st.markdown(f"Télécharger le modèle [ici](./{filename})")
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

            # Mettre à jour l'état de session avec trained_models
            st.session_state['trained_models'] = trained_models

        st.header("Téléchargement des Modèles Entraînés")

        # Afficher les boutons de téléchargement pour chaque modèle entraîné
        for model_id, model in trained_models.items():
            filename = f"modele_entraine_{model_name}_{model_id}.pkl"
            st.download_button(
                label=f"Télécharger le Modèle {model_id}",
                data=pickle.dumps(model),
                file_name=filename,
                mime="application/octet-stream"
            )

    else:
        st.warning("Aucun dataframe trouvé dans l'état de session. Veuillez d'abord charger ou connecter les données.")

if __name__ == "__main__":
    main()
