import streamlit as st
import pandas as pd
import mysql.connector
from pymongo import MongoClient


def main():
    """
    Fonction principale pour la page de connexion aux données.
    """
    st.title("Page de Connexion aux Données")
    st.write("Ceci est la page de connexion aux données.")

    option = st.selectbox("Sélectionner la Source de Données", ["Télécharger CSV/Excel", "Se Connecter à une Base de "
                                                                                         "Données"])

    if option == "Télécharger CSV/Excel":
        # Section pour télécharger un fichier CSV ou Excel
        uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['dataframe'] = df
                st.session_state['original_dataframe'] = df.copy()
                st.dataframe(df)
            except Exception as e:
                st.error(f"Erreur: {e}")

    elif option == "Se Connecter à une Base de Données":
        # Section pour se connecter à une base de données
        db_type = st.selectbox("Sélectionner le Type de Base de Données", ["MySQL", "MongoDB"])

        if db_type == "MySQL":
            # Connexion à MySQL
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
                    cursor = conn.cursor()
                    cursor.execute("SHOW TABLES")
                    tables = [table[0] for table in cursor.fetchall()]
                    selected_table = st.selectbox("Sélectionner la Table", tables)
                    if selected_table:
                        query = f"SELECT * FROM {selected_table}"
                        df = pd.read_sql(query, conn)
                        st.session_state['dataframe'] = df
                        st.session_state['original_dataframe'] = df.copy()
                        st.dataframe(df)
                except Exception as e:
                    st.error(f"Erreur: {e}")

        elif db_type == "MongoDB":
            # Connexion à MongoDB
            mongo_host = st.text_input("Hôte MongoDB", "localhost")
            mongo_port = st.text_input("Port MongoDB", "27017")
            mongo_dbname = st.text_input("Base de Données MongoDB")

            if st.button("Se Connecter à MongoDB"):
                try:
                    client = MongoClient(f"mongodb://{mongo_host}:{mongo_port}/")
                    st.session_state['mongo_client'] = client
                    st.session_state['mongo_db'] = client[mongo_dbname]
                    st.session_state['connected'] = True
                except Exception as e:
                    st.error(f"Erreur: {e}")

            if st.session_state.get('connected'):
                # Sélectionner une collection MongoDB
                db = st.session_state['mongo_db']
                collections = db.list_collection_names()
                selected_collection = st.selectbox("Sélectionner la Collection", collections)
                if selected_collection:
                    collection = db[selected_collection]
                    df = pd.DataFrame(list(collection.find()))
                    if "_id" in df.columns:
                        df = df.drop(columns=["_id"])
                    st.session_state['dataframe'] = df
                    st.session_state['original_dataframe'] = df.copy()
                    st.dataframe(df)


if __name__ == "__main__":
    if 'connected' not in st.session_state:
        st.session_state['connected'] = False
    main()
