import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import mysql.connector
from pymongo import MongoClient
import io

def main():
    st.title("Page de Connexion aux Données")
    st.write("Ceci est la page de connexion aux données.")

    option = st.selectbox("Sélectionner la Source de Données",
                          ["Télécharger CSV/Excel", "Se Connecter à une Base de Données"])

    if option == "Télécharger CSV/Excel":
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
                st.error(f"Erreur lors du chargement du fichier : {e}")

    elif option == "Se Connecter à une Base de Données":
        db_type = st.selectbox("Sélectionner le Type de Base de Données", ["PostgreSQL", "MySQL", "MongoDB"])

        if db_type == "MySQL":
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

        elif db_type == "PostgreSQL":
            postgres_host = st.text_input("Hôte PostgreSQL", "197.140.18.127")
            postgres_port = st.text_input("Port PostgreSQL", "6432")
            postgres_user = st.text_input("Utilisateur PostgreSQL", "salam_report")
            postgres_password = st.text_input("Mot de passe PostgreSQL", type="password")
            postgres_database = st.text_input("Base de Données PostgreSQL", "dbsalamprod")

            if st.button("Se Connecter à PostgreSQL"):
                try:
                    engine = create_engine(
                        f'postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}')
                    conn = engine.connect()
                    st.session_state['postgres_conn'] = conn

                    result = conn.execute(
                        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
                    tables = [row[0] for row in result]
                    st.session_state['tables'] = tables

                    st.write("Connexion réussie et tables chargées.")
                    st.write(f"Tables disponibles : {tables}")
                except Exception as e:
                    st.error(f"Erreur lors de la connexion PostgreSQL : {e}")

        if 'postgres_conn' in st.session_state and st.session_state['postgres_conn']:
            st.write("Connexion PostgreSQL établie.")
            tables = st.session_state.get('tables', [])

            if tables:
                selected_tables = st.multiselect("Sélectionner les Tables", tables)

                if len(selected_tables) == 2:
                    try:
                        conn = st.session_state['postgres_conn']
                        dfs = {}
                        for table in selected_tables:
                            query = text(f'SELECT * FROM "{table}"')
                            dfs[table] = pd.read_sql(query, conn)

                        st.write("Sélectionner les colonnes pour chaque table :")
                        selected_columns = {}
                        for table, df in dfs.items():
                            columns = st.multiselect(f"Colonnes de {table}", df.columns.tolist(),
                                                     default=df.columns.tolist())
                            selected_columns[table] = columns

                        merge_key_table1 = st.selectbox(f"Clé de fusion dans {selected_tables[0]}",
                                                        dfs[selected_tables[0]].columns.tolist())
                        merge_key_table2 = st.selectbox(f"Clé de fusion dans {selected_tables[1]}",
                                                        dfs[selected_tables[1]].columns.tolist())

                        if merge_key_table1 and merge_key_table2:
                            merged_df = pd.merge(
                                dfs[selected_tables[0]][selected_columns[selected_tables[0]]],
                                dfs[selected_tables[1]][selected_columns[selected_tables[1]]],
                                left_on=merge_key_table1,
                                right_on=merge_key_table2,
                                how='inner'
                            )
                            st.session_state['dataframe'] = merged_df
                            st.session_state['original_dataframe'] = merged_df.copy()

                            st.write("Sélectionner les colonnes pour la table fusionnée :")
                            all_columns = merged_df.columns.tolist()
                            selected_merge_columns = st.multiselect("Colonnes de la table fusionnée", all_columns,
                                                                    default=all_columns)

                            if selected_merge_columns:
                                final_df = merged_df[selected_merge_columns]
                                st.session_state['final_dataframe'] = final_df
                                st.dataframe(final_df)
                            else:
                                st.write("Veuillez sélectionner au moins une colonne.")
                            st.write("Table fusionnée:")
                            st.dataframe(final_df)

                            # Ajouter un bouton de téléchargement pour le DataFrame fusionné
                            csv = final_df.to_csv(index=False).encode('utf-8')

                            # Boutons de téléchargement
                            st.download_button(
                                label="Télécharger le tableau nettoyé en CSV",
                                data=csv,
                                file_name='merged_data.csv',
                                mime='text/csv'
                            )

                            towrite = io.BytesIO()
                            final_df.to_excel(towrite, index=False, engine='xlsxwriter')
                            towrite.seek(0)
                            st.download_button(
                                label="Télécharger le tableau nettoyé en Excel",
                                data=towrite,
                                file_name='merged_data.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                        else:
                            st.error("Veuillez sélectionner les colonnes de fusion dans les deux tables.")
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture des tables ou de la fusion : {e}")
            else:
                st.write("Aucune table trouvée dans la base de données.")

if __name__ == "__main__":
    if 'dataframe' not in st.session_state:
        st.session_state['dataframe'] = None
    if 'original_dataframe' not in st.session_state:
        st.session_state['original_dataframe'] = None
    if 'final_dataframe' not in st.session_state:
        st.session_state['final_dataframe'] = None
    if 'tables' not in st.session_state:
        st.session_state['tables'] = []
    if 'postgres_conn' not in st.session_state:
        st.session_state['postgres_conn'] = None
    main()
