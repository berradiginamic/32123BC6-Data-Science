import streamlit as st
import pandas as pd


def detect_na(df):
    if df is not None:
        na_columns = df.columns[df.isna().any()]
        if len(na_columns) > 0:
            st.write("Colonnes avec des valeurs manquantes :")
            for column in na_columns:
                na_count = df[column].isna().sum()
                st.write(f"{column}: {na_count} valeurs manquantes")
        else:
            st.write("Pas de valeurs manquantes trouvées dans le jeu de données.")
    else:
        st.warning("Le DataFrame est vide.")


# Function to load data
def load_data():
    uploaded_file = st.sidebar.file_uploader("Charger le fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu du jeu de données :")
        st.write(df.head())

        # Print total rows and columns
        st.text(f"Nombre total de colonnes : {df.shape[1]}")
        st.text(f"Nombre total de lignes : {df.shape[0]}")

        return df
    else:
        st.warning("Veuillez charger un fichier CSV pour continuer.")
        return None


def count_unique_values(df):
    if df is not None:
        st.write("Nombre de valeurs uniques dans chaque colonne :")
        unique_counts = {}
        for column in df.columns:
            unique_counts[column] = df[column].nunique()
            st.write(f"{column}: {unique_counts[column]}")
    else:
        st.warning("Le DataFrame est vide.")


def show_data_processing_options(df):
    st.sidebar.subheader("Étape 3: Traitement des données")

    if df is not None:
        selected_columns = st.sidebar.multiselect("Sélectionnez les features", df.columns, key='a')
        target_column = st.sidebar.selectbox("Sélectionnez la target", df.columns)
        return selected_columns, target_column
    else:
        return None, None


def descriptive_analysis(df):
    if st.sidebar.checkbox("Analyse descriptive"):
        st.write("Statistiques descriptives du jeu de données :")
        st.write(df.describe())