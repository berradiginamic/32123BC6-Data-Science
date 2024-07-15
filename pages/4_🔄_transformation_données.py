import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def main():
    """
    Fonction principale pour la page de Transformation des Données.
    Permet de gérer les données manquantes, effectuer des transformations, encoder les variables catégorielles,
    gérer les données de date et heure, détecter et traiter les valeurs aberrantes, gérer les types de données,
    sauvegarder et réinitialiser les données traitées.
    """
    st.title("Page de Transformation des Données")
    st.write("Cette page fournit des fonctionnalités de traitements des données.")

    # Vérifier si le dataframe est disponible dans l'état de session
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        st.dataframe(df)

        # Gestion des Données Manquantes
        st.header("Gestion des Données Manquantes")
        if st.checkbox("Afficher les Données Manquantes"):
            st.write(df.isnull().sum())

        missing_data_method = st.selectbox("Gérer les Données Manquantes",
                                           ["Aucune", "Supprimer les lignes", "Supprimer les colonnes", "Remplir avec la Moyenne",
                                            "Remplir avec la Médiane", "Remplir avec le Mode", "Remplir avec une Valeur Spécifique"])

        if missing_data_method == "Supprimer les lignes":
            df = df.dropna()
        elif missing_data_method == "Supprimer les colonnes":
            df = df.dropna(axis=1)
        elif missing_data_method == "Remplir avec la Moyenne":
            df = df.fillna(df.mean())
        elif missing_data_method == "Remplir avec la Médiane":
            df = df.fillna(df.median())
        elif missing_data_method == "Remplir avec le Mode":
            df = df.fillna(df.mode().iloc[0])
        elif missing_data_method == "Remplir avec une Valeur Spécifique":
            fill_value = st.text_input("Entrer la valeur pour remplir les données manquantes")
            if fill_value:
                df = df.fillna(fill_value)

        st.dataframe(df)

        # Transformation des Données
        st.header("Transformation des Données")
        transformation_method = st.selectbox("Sélectionner la Méthode de Transformation",
                                             ["Aucune", "Normalisation (Min-Max)", "Standardisation (Z-score)"])

        if transformation_method == "Normalisation (Min-Max)":
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))
        elif transformation_method == "Standardisation (Z-score)":
            scaler = StandardScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))

        st.dataframe(df)

        # Encodage des Variables Catégorielles
        st.header("Encodage des Variables Catégorielles")
        encoding_method = st.selectbox("Sélectionner la Méthode d'Encodage", ["Aucun", "Encodage One-Hot", "Encodage Label"])

        if encoding_method == "Encodage One-Hot":
            df = pd.get_dummies(df)
        elif encoding_method == "Encodage Label":
            le = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = le.fit_transform(df[col])

        st.dataframe(df)

        # Filtrage et Sélection des Données
        st.header("Filtrage et Sélection des Données")
        if st.checkbox("Supprimer les Doublons"):
            df = df.drop_duplicates()

        st.dataframe(df)

        columns_to_drop = st.multiselect("Sélectionner les Colonnes à Supprimer", df.columns)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        st.dataframe(df)

        # Gestion des Données de Date et Heure
        st.header("Gestion des Données de Date et Heure")
        date_columns = st.multiselect("Sélectionner les Colonnes de Date", df.columns[df.dtypes == 'object'])

        for date_col in date_columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[f'{date_col}_jour'] = df[date_col].dt.day
            df[f'{date_col}_mois'] = df[date_col].dt.month
            df[f'{date_col}_année'] = df[date_col].dt.year

        st.dataframe(df)

        # Gestion des Valeurs Aberrantes
        st.header("Gestion des Valeurs Aberrantes")
        outlier_method = st.selectbox("Sélectionner la Méthode de Détection des Valeurs Aberrantes", ["Aucune", "Z-score", "IQR"])

        if outlier_method == "Z-score":
            z_scores = np.abs((df - df.mean()) / df.std())
            df = df[(z_scores < 3).all(axis=1)]
        elif outlier_method == "IQR":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        st.dataframe(df)

        # Gestion des Types de Données
        st.header("Gestion des Types de Données")
        for col in df.columns:
            col_type = st.selectbox(f"Sélectionner le type pour la colonne {col}", ["Aucun", "int", "float", "str"], index=0)
            if col_type != "Aucun":
                df[col] = df[col].astype(col_type)

        st.dataframe(df)

        # Sauvegarde des Données Traitées / Réinitialisation du DataFrame
        st.header("Sauvegarde/Réinitialisation des Données Traitées")
        if st.button("Sauvegarder les Données Traitées"):
            st.session_state['dataframe'] = df
            st.success("Données traitées sauvegardées et mises à jour!")

        if st.button("Réinitialiser le DataFrame"):
            st.session_state['dataframe'] = st.session_state['original_dataframe']
            st.success("DataFrame réinitialisé à l'état d'origine.")

    else:
        st.write("Aucune donnée disponible. Veuillez télécharger ou vous connecter à une source de données sur la page de connexion aux données.")


if __name__ == "__main__":
    main()
