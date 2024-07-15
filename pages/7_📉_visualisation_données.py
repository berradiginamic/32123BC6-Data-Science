import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    st.title("Page de Visualisation des Données")
    st.write("Ceci est la page de visualisation des données.")

    # Vérifier si le dataframe est disponible dans l'état de session
    if 'dataframe' in st.session_state and st.session_state['dataframe'] is not None:
        df = st.session_state['dataframe']

        st.header("Sélection des Options de Visualisation")

        # Sélectionner le type de visualisation
        plot_types = st.multiselect("Sélectionner le Type de Visualisation",
                                    ["Histogramme", "Diagramme à Barres", "Diagramme de Compte", "Nuage de Points", "Boîte à Moustaches",
                                     "Carte de Chaleur de Corrélation"])

        # Sélectionner les colonnes pour la visualisation
        columns = df.columns.tolist()
        columns_to_visualize = st.multiselect("Sélectionner les Colonnes pour la Visualisation", columns)

        # Sélectionner la colonne cible pour des tracés spécifiques (par exemple, nuage de points)
        target_column = st.selectbox("Sélectionner la Colonne Cible (pour le Nuage de Points)", columns)

        # Générer les tracés sélectionnés
        if "Histogramme" in plot_types:
            st.header("Histogramme")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                ax.set_title(f"Histogramme de {column}")
                st.pyplot(fig)

        if "Diagramme à Barres" in plot_types:
            st.header("Diagramme à Barres")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.countplot(x=column, data=df, ax=ax)
                ax.set_title(f"Diagramme à Barres de {column}")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

        if "Diagramme de Compte" in plot_types:
            st.header("Diagramme de Compte")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.countplot(y=column, data=df, ax=ax)
                ax.set_title(f"Diagramme de Compte de {column}")
                st.pyplot(fig)

        if "Nuage de Points" in plot_types and target_column:
            st.header("Nuage de Points")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[column], y=df[target_column], ax=ax)
                ax.set_title(f"Nuage de Points de {column} vs {target_column}")
                st.pyplot(fig)

        if "Boîte à Moustaches" in plot_types:
            st.header("Boîte à Moustaches")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.boxplot(x=column, data=df, ax=ax)
                ax.set_title(f"Boîte à Moustaches de {column}")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

        if "Carte de Chaleur de Corrélation" in plot_types:
            st.header("Carte de Chaleur de Corrélation")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            ax.set_title("Carte de Chaleur de Corrélation")
            st.pyplot(fig)

    else:
        st.write("Aucune donnée disponible. Veuillez télécharger ou vous connecter à une source de données sur la page de Connexion aux Données.")


if __name__ == "__main__":
    main()
