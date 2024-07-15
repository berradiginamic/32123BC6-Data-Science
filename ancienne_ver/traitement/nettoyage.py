# In traitement/data_processing.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder, LabelEncoder


# Function to apply encoding methods to categorical columns
def apply_encoding_methods(df, encoding_methods):
    encoded_df = df.copy()
    for method in encoding_methods:
        if method == 'One-Hot Encoding':
            # Apply One-Hot Encoding to categorical columns
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            encoded_data = encoder.fit_transform(df.select_dtypes(include='object'))
            encoded_columns = encoder.get_feature_names_out(df.select_dtypes(include='object').columns)
            encoded_df = pd.concat([encoded_df, pd.DataFrame(encoded_data, columns=encoded_columns)], axis=1)
        elif method == 'Label Encoding':
            # Apply Label Encoding to categorical columns
            encoder = LabelEncoder()
            for column in df.select_dtypes(include='object').columns:
                encoded_df[column] = encoder.fit_transform(df[column])
        # Add more encoding methods as needed
    return encoded_df

def apply_normalization(df, normalization_technique):
    # Check if the DataFrame contains numeric values
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    if len(numeric_columns) == 0:
        raise ValueError("DataFrame does not contain numeric columns for normalization")

    # Filter out non-numeric columns
    numeric_df = df[numeric_columns]
    if normalization_technique == 'StandardScaler':
        # Instantiate StandardScaler
        scaler = StandardScaler()
        # Fit and transform the data
        normalized_data = scaler.fit_transform(numeric_df)
    elif normalization_technique == 'MinMaxScaler':
        # Instantiate MinMaxScaler
        scaler = MinMaxScaler()
        # Fit and transform the data
        normalized_data = scaler.fit_transform(numeric_df)
    elif normalization_technique == 'RobustScaler':
        # Instantiate RobustScaler
        scaler = RobustScaler()
        # Fit and transform the data
        normalized_data = scaler.fit_transform(numeric_df)
    elif normalization_technique == 'Normalizer':
        # Instantiate Normalizer
        scaler = Normalizer()
        # Fit and transform the data
        normalized_data = scaler.fit_transform(numeric_df)
    elif normalization_technique == 'Log Transformation':
        # Perform Log Transformation
        normalized_data = numeric_df.apply(lambda x: np.log(x + 1))
    elif normalization_technique == 'Normalization by Division':
        # Perform Normalization by Division
        normalized_data = numeric_df.div(df.sum(axis=1), axis=0)
    else:
        raise ValueError("Invalid normalization technique")

    # Convert the scaled data back to a DataFrame (optional)
    """
    normalized_df entraine l'erreur de 'Shape of passed values is (xxx)' 
    """
    #normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
    # Reconstruct the DataFrame with normalized data
    #normalized_df = pd.DataFrame(normalized_data, columns=df.select_dtypes(include='number').columns, index=df.index)
    # Concatenate non-numeric columns (if any)
    #non_numeric_columns = df.columns.difference(numeric_columns)
    #for column in non_numeric_columns:
    #    numeric_df[column] = df[column]
    return numeric_df


def drop_empty_columns(df):
    # Check if there are any empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()

    # Drop empty columns if any
    if empty_columns:
        df = df.drop(columns=empty_columns)
        st.write(f"Colonnes vides supprimées : {empty_columns}")
    else:
        st.write("Aucune colonne vide trouvée.")
    return df


def detect_and_drop_similar_to_index_columns(df):
    similar_columns = []
    for column in df.columns:
        if df[column].equals(df.index.to_series()):
            similar_columns.append(column)
    if similar_columns:
        st.write("Colonnes similaires à l'index détectées :")
        st.write(similar_columns)
        if st.checkbox('Supprimer les colonnes similaires à l\'index'):
            df = df.drop(columns=similar_columns)
            st.write("Colonnes similaires à l'index supprimées avec succès.")
    else:
        st.write("Aucune colonne similaire à l'index trouvée.")
    return df


def detect_outliers(df, threshold=3):
    outliers = pd.DataFrame()
    for column in df.select_dtypes(include=['float', 'int']).columns:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers[column] = df[column][abs(z_scores) > threshold]
    return outliers


def treat_outliers(df, threshold=3):
    for column in df.select_dtypes(include=['float', 'int']).columns:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df[column][z_scores > threshold] = df[column].mean()
    return df


def correlation_with_target(df, target_column):
    if st.sidebar.checkbox("Corrélation avec la cible"):
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = df[numeric_columns].corr()
        st.write("Matrice de corrélation :")
        st.write(correlation_matrix)


def handle_missing_values(df):
    # User options for handling missing values
    # User options for handling missing values
    na_option = st.selectbox("Choisir l'action à effectuer pour les valeurs manquantes :",
                             ['Ne rien faire', 'Supprimer les lignes', 'Supprimer les colonnes', 'Imputer'])

    if na_option == 'Supprimer les lignes':
        df = df.dropna(axis=0)
    elif na_option == 'Supprimer les colonnes':
        df = df.dropna(axis=1)
    elif na_option == 'Imputer':
        impute_method = st.selectbox("Choisir la méthode d'imputation :", ['Moyenne', 'Médiane', 'Mode'])
        method_map = {'Moyenne': 'mean', 'Médiane': 'median', 'Mode': 'most_frequent'}
        axis = 0 if st.checkbox('Appliquer aux lignes') else 1
        method = method_map[impute_method]

        # Identify numerical columns for imputation
        numerical_columns = df.select_dtypes(include=['number']).columns
        if axis == 0:
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
        else:
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    return df


def column_frequencies(df, target_column):
    if st.sidebar.checkbox("Fréquences"):
        st.write("Fréquence des valeurs dans la colonne cible :")
        st.write(df[target_column].value_counts())


def standardization(df):
    if st.sidebar.checkbox("Standardisation"):
        st.write("Standardisation des colonnes numériques :")
        numeric_columns = df.select_dtypes(include=['float64']).columns
        df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
        st.write(df[numeric_columns].head())