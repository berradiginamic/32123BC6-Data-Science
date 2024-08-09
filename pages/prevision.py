import streamlit as st
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
import numpy as np
import matplotlib.pyplot as plt

# Function to connect to MySQL
def connect_to_mysql():
    mysql_host = st.text_input("Hôte MySQL", "localhost")
    mysql_user = st.text_input("Utilisateur MySQL", "root")
    mysql_password = st.text_input("Mot de passe MySQL", type="password")
    mysql_database = st.text_input("Base de Données MySQL")

    if st.button("Se Connecter à MySQL"):
        try:
            conn = mysql.connector.connect(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database
            )
            st.session_state['mysql_conn'] = conn
            st.session_state['data_loaded'] = False
            st.success("Connexion réussie à MySQL!")
        except Exception as e:
            st.error(f"Erreur: {e}")

# Function to connect to PostgreSQL
def connect_to_postgresql():
    postgres_host = st.text_input("Hôte PostgreSQL", "197.140.18.127")
    postgres_port = st.text_input("Port PostgreSQL", "6432")
    postgres_user = st.text_input("Utilisateur PostgreSQL", "salam_report")
    postgres_password = st.text_input("Mot de passe PostgreSQL", type="password")
    postgres_database = st.text_input("Base de Données PostgreSQL", "dbsalamprod")

    if st.button("Se Connecter à PostgreSQL"):
        try:
            engine = create_engine(
                f'postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}'
            )
            st.session_state['postgres_conn'] = engine
            st.session_state['data_loaded'] = False
            st.success("Connexion réussie à PostgreSQL!")
        except Exception as e:
            st.error(f"Erreur lors de la connexion PostgreSQL : {e}")

# Function to connect to MongoDB
def connect_to_mongodb():
    mongo_host = st.text_input("Hôte MongoDB", "localhost")
    mongo_port = st.text_input("Port MongoDB", "27017")
    mongo_dbname = st.text_input("Base de Données MongoDB")

    if st.button("Se Connecter à MongoDB"):
        try:
            client = MongoClient(f"mongodb://{mongo_host}:{mongo_port}/")
            st.session_state['mongo_client'] = client
            st.session_state['mongo_db'] = client[mongo_dbname]
            st.session_state['data_loaded'] = False
            st.success("Connexion réussie à MongoDB!")
        except Exception as e:
            st.error(f"Erreur: {e}")

# Function to load data with transaction handling
def load_data_with_transaction_handling(engine, query):
    """
    Loads data from the database with transaction handling.
    Returns a DataFrame.
    """
    try:
        with engine.begin() as connection:
            df = pd.read_sql(query, connection)
            st.session_state['dataframe'] = df
            st.session_state['original_dataframe'] = df.copy()
            st.session_state['data_loaded'] = True
            st.write("Aperçu des données :")
            st.write(df.head())
            st.success("Données chargées avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")

# Function to encode categorical columns using Label Encoding
def encode_categorical_columns(df):
    """
    Encode categorical columns using Label Encoding
    """
    le_dict = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            le_dict[column] = le
    return df, le_dict

# Function to build an Artificial Neural Network model
def build_ann(input_dim):
    """
    Build an Artificial Neural Network model
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to build a Recurrent Neural Network model
def build_rnn(input_dim):
    """
    Build a Recurrent Neural Network model
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(input_dim, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to build a Convolutional Neural Network model
def build_cnn(input_dim):
    """
    Build a Convolutional Neural Network model
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to calculate category probabilities for each customer
def calculate_category_probabilities(df, category_level):
    """
    Calculate category probabilities for each customer.
    """
    if category_level not in df.columns:
        st.error(f"Colonne {category_level} non trouvée dans les données.")
        return pd.DataFrame()

    probabilities = df.groupby('client')[category_level].value_counts(normalize=True).unstack().fillna(0)
    return probabilities

# Function to visualize category probabilities for a given customer
def visualize_probabilities(probabilities, customer_id):
    """
    Visualize category probabilities for a given customer.
    """
    if customer_id not in probabilities.index:
        st.error(f"ID de client {customer_id} non trouvé dans les données.")
        return

    customer_probs = probabilities.loc[customer_id]
    st.write(f"Probabilités des catégories pour le client {customer_id}:")
    st.bar_chart(customer_probs)

# Main function to run the Streamlit app
def main():
    st.title("Application Générale de Prévision")
    st.write("Cette application permet de faire des prévisions basées sur des données SQL personnalisées.")

    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False

    # Display connection options first
    if not st.session_state['data_loaded']:
        st.subheader("Connexion à la Base de Données ou Chargement de Fichier")
        option = st.selectbox("Sélectionner la source des données",
                              ["Se Connecter à une Base de Données", "Télécharger CSV/Excel"])

        if option == "Se Connecter à une Base de Données":
            db_type = st.selectbox("Sélectionner le Type de Base de Données", ["PostgreSQL", "MySQL", "MongoDB"])

            if db_type == "MySQL":
                connect_to_mysql()
            elif db_type == "PostgreSQL":
                connect_to_postgresql()
            elif db_type == "MongoDB":
                connect_to_mongodb()

            if 'mysql_conn' in st.session_state:
                conn = st.session_state['mysql_conn']
                query = st.text_area("Entrez votre requête SQL ici", "SELECT * FROM your_table LIMIT 10;")
                if st.button("Exécuter la Requête"):
                    load_data_with_transaction_handling(conn, query)

            elif 'postgres_conn' in st.session_state:
                engine = st.session_state['postgres_conn']
                query = st.text_area("Entrez votre requête SQL ici", "SELECT * FROM your_table LIMIT 10;")
                if st.button("Exécuter la Requête"):
                    load_data_with_transaction_handling(engine, query)

            elif 'mongo_client' in st.session_state:
                db = st.session_state['mongo_db']
                collections = db.list_collection_names()
                selected_collection = st.selectbox("Sélectionner la Collection", collections)
                if selected_collection:
                    collection = db[selected_collection]
                    df = pd.DataFrame(list(collection.find()))
                    st.session_state['dataframe'] = df
                    st.session_state['original_dataframe'] = df.copy()
                    st.session_state['data_loaded'] = True
                    st.write("Aperçu des données :")
                    st.dataframe(df)

        elif option == "Télécharger CSV/Excel":
            uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv", "xlsx"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state['dataframe'] = df
                    st.session_state['original_dataframe'] = df.copy()
                    st.session_state['data_loaded'] = True
                    st.write("Aperçu des données :")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Erreur lors du chargement du fichier : {e}")

    # Only show further options if data is loaded
    if st.session_state['data_loaded']:
        st.subheader("Chargement et Nettoyage des Données")

        # Data cleaning options
        df = st.session_state['dataframe']

        if st.checkbox("Supprimer les doublons"):
            df = df.drop_duplicates()
            st.session_state['dataframe'] = df

        if st.checkbox("Supprimer les valeurs manquantes"):
            df = df.dropna()
            st.session_state['dataframe'] = df

        if st.checkbox("Remplacer les valeurs manquantes par la moyenne"):
            # Apply only on numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.session_state['dataframe'] = df

        # Display the cleaned data
        st.write("Aperçu des données nettoyées :")
        st.dataframe(st.session_state['dataframe'])

        st.subheader("Prétraitement des Données")

        # Display the encoded data
        df_encoded, le_dict = encode_categorical_columns(st.session_state['dataframe'].copy())
        st.write("Aperçu des données encodées :")
        st.dataframe(df_encoded)

        # Use a selectbox to allow column selection for the target variable
        target_column = st.selectbox("Sélectionner la Colonne Cible", df.columns.tolist())

        # Validate target column and preprocess
        if target_column:
            if target_column in df_encoded.columns:
                # Prepare data for modeling
                X = df_encoded.drop(target_column, axis=1).values
                y = df_encoded[target_column].values

                # Normalize data
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)

                # Split data into training and test sets
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Store preprocessed data in session
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['preprocessed'] = True
                st.write("Données prétraitées.")
            else:
                st.error(f"La colonne '{target_column}' n'existe pas dans les données encodées.")
        else:
            st.warning("Veuillez sélectionner la colonne cible.")

        # Show modeling options if data is preprocessed
        if 'preprocessed' in st.session_state and st.session_state['preprocessed']:
            st.subheader("Modélisation avec Réseaux de Neurones")
            model_type = st.selectbox("Sélectionner le Type de Modèle", ["ANN", "RNN", "CNN"])
            input_dim = st.session_state['X_train'].shape[1]
            model = None

            if model_type == "ANN":
                model = build_ann(input_dim)
            elif model_type == "RNN":
                model = build_rnn(input_dim)
            elif model_type == "CNN":
                model = build_cnn(input_dim)

            if st.button("Entraîner le Modèle"):
                X_train = np.array(st.session_state['X_train'])
                y_train = np.array(st.session_state['y_train'])
                X_test = np.array(st.session_state['X_test'])
                y_test = np.array(st.session_state['y_test'])

                # Train the model and capture the history
                history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
                st.session_state['model'] = model
                st.success("Modèle entraîné avec succès.")

                # Plot the training and validation loss
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_title('Training and Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)

                # Perform predictions
                train_predict = model.predict(X_train)
                test_predict = model.predict(X_test)

                # Display predictions
                st.write("Prédictions terminées. Voici un aperçu des résultats :")
                st.write("Prédictions d'entraînement :")
                st.dataframe(train_predict)
                st.write("Prédictions de test :")
                st.dataframe(test_predict)

            # Visualize category probabilities
            if st.checkbox("Calculer et Visualiser les Probabilités de Catégories"):
                # Debug: List all column names
                st.write("Colonnes disponibles pour les niveaux de catégories:")
                st.write(df.columns)

                # Updated logic to match 'level' with lowercase
                category_level_choices = [col for col in df.columns if col.startswith('level')]
                if category_level_choices:
                    category_level = st.selectbox("Sélectionner le niveau de catégorie", category_level_choices)

                    df['customer_id'] = df['client']  # Ensure client ID is the correct column
                    probabilities = calculate_category_probabilities(df, category_level)

                    if not probabilities.empty:
                        customer_id = st.selectbox("Sélectionner l'ID du Client", probabilities.index)
                        visualize_probabilities(probabilities, customer_id)
                else:
                    st.warning("Aucun niveau de catégorie trouvé dans les données.")

if __name__ == "__main__":
    main()
