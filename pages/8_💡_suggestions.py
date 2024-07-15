import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder


def main():
    """
    Fonction principale pour la page de conseils de nettoyage des données.
    Propose des suggestions de nettoyage basées sur l'ensemble de données stocké dans l'état de session.
    """
    st.title("Conseils pour le Nettoyage des Données")
    st.write("Cette page fournit des suggestions de nettoyage des données basées sur votre ensemble de données.")

    # Vérifier si le dataframe est disponible dans l'état de session
    if 'dataframe' in st.session_state and st.session_state['dataframe'] is not None:
        df = st.session_state['dataframe']

        st.header("Suggestions de Nettoyage des Données")

        # 1. Analyse de Corrélation
        st.subheader("Analyse de Corrélation")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))  # Créer une figure et un axe
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title("Carte de Chaleur de Corrélation")
        st.pyplot(fig)  # Afficher le tracé à l'aide de st.pyplot()

        high_corr_pairs = [(corr.columns[i], corr.columns[j]) for i, j in zip(*np.where(corr > 0.8)) if i != j]
        if high_corr_pairs:
            st.write("Colonnes fortement corrélées (> 0.8) :")
            for pair in high_corr_pairs:
                st.write(f"- {pair}")

        # 2. Détection des Données Manquantes
        st.subheader("Détection des Données Manquantes")
        missing_data = df.isnull().mean() * 100
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if not missing_data.empty:
            st.write("Colonnes avec des données manquantes (%):")
            st.write(missing_data)

        # 3. Détection des Valeurs Aberrantes (exemple)
        st.subheader("Détection des Valeurs Aberrantes")
        numerical_cols = df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            # Exemple : Méthode du Z-score pour la détection des valeurs aberrantes
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[(z_scores > 3)]
            if not outliers.empty:
                st.write(f"Valeurs aberrantes détectées dans {col} :")
                st.write(outliers.head())

        # 4. Types de Données
        st.subheader("Types de Données")
        data_types = df.dtypes
        st.write("Types de données actuels :")
        st.write(data_types)

        # Exemple : Détection des colonnes catégorielles et suggestion d'encodage par étiquette
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            st.write("Colonnes catégorielles détectées :")
            st.write(categorical_cols)

            # Exemple : Suggestion d'encodage par étiquette
            le = LabelEncoder()
            encoded = df[categorical_cols].apply(le.fit_transform)
            st.write("Exemple d'encodage par étiquette :")
            st.write(encoded.head())

        # 5. Lignes en Double
        st.subheader("Lignes en Double")
        duplicate_rows = df[df.duplicated()]
        if not duplicate_rows.empty:
            st.write("Lignes en double détectées :")
            st.write(duplicate_rows.head())

        # 6. Colonnes ID Potentielles
        st.subheader("Colonnes ID Potentielles")
        id_columns = []
        for col in df.columns:
            if df[col].nunique() == df.shape[0]:  # Toutes les valeurs sont uniques
                id_columns.append(col)
        if id_columns:
            st.write("Colonnes ID potentielles détectées :")
            st.write(id_columns)

    else:
        st.write("Aucune donnée disponible. Veuillez télécharger ou vous connecter à une source de données sur la page de Connexion aux Données.")


if __name__ == "__main__":
    main()
