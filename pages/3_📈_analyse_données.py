import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def main():
    """
    Fonction principale pour la page d'Analyse des Données.
    Réalise l'Analyse Exploratoire des Données (AED) et les analyses statistiques sur un jeu de données.
    """
    st.title("Page d'Analyse des Données")
    st.write("Cette page fournit des insights d'Analyse Exploratoire des Données (AED) et des analyses statistiques.")

    # Vérifier si le dataframe est disponible dans l'état de session
    if 'dataframe' in st.session_state and st.session_state['dataframe'] is not None:
        df = st.session_state['dataframe']

        st.header("Statistiques Sommaires")
        st.write("Statistiques de base du jeu de données :")
        st.write(df.describe())
        st.write("""
        **Explication** : Les statistiques sommaires fournissent un aperçu rapide des caractéristiques numériques du jeu de données, y compris des mesures telles que la moyenne, l'écart-type et les quartiles.
        Elles aident à comprendre la tendance centrale, la dispersion et la plage de valeurs pour chaque colonne numérique.
        """)

        st.header("Décompte des Valeurs")
        st.write("Décompte des valeurs pour les variables catégorielles :")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            st.subheader(col)
            st.write(df[col].value_counts())
            st.write(f"""
            **Explication** : Les décomptes des valeurs montrent la fréquence de chaque catégorie dans les variables catégorielles. 
            Ils aident à comprendre la distribution et le déséquilibre des catégories, ce qui est important pour l'ingénierie des caractéristiques et les décisions de modélisation.
            """)

        st.header("Analyse Statistique")

        # Exemple : Test d'Hypothèse (exemple de test t)
        st.subheader("Test d'Hypothèse (Exemple : test t)")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Sélectionner la première colonne numérique pour le test t", numeric_cols, index=0)
            col2 = st.selectbox("Sélectionner la deuxième colonne numérique pour le test t", numeric_cols, index=1)
            if st.button("Exécuter le test t"):
                t_stat, p_value = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
                st.write(f"Résultat du test t entre {col1} et {col2} :")
                st.write(f"Statistique t : {t_stat}")
                st.write(f"Valeur de p : {p_value}")
                st.write("""
                **Explication** : Le test t compare les moyennes de deux groupes de données numériques pour déterminer s'ils sont significativement différents l'un de l'autre.
                La statistique t mesure la différence par rapport à la variation des données, tandis que la valeur de p indique la signification de la différence.
                """)

                if p_value < 0.05:
                    st.write("Il y a une différence significative entre les groupes.")
                else:
                    st.write("Il n'y a pas de différence significative entre les groupes.")
                st.write("""
                **Interprétation** : Si la valeur de p est inférieure à 0,05, cela suggère que la différence observée entre les groupes n'est probablement pas due au hasard (c'est-à-dire statistiquement significative).
                Cela pourrait impliquer une différence significative entre les variables comparées.
                """)
        else:
            st.write("Nombre insuffisant de colonnes numériques disponibles pour le test t. Il faut au moins deux colonnes numériques.")
            st.write("""
            **Explication** : Le test t nécessite au moins deux colonnes numériques avec suffisamment de points de données pour la comparaison. 
            Veuillez vous assurer d'avoir suffisamment de colonnes numériques pour effectuer ce test.
            """)

        # Exemple : Analyse de Corrélation
        st.subheader("Analyse de Corrélation")
        st.write("Matrice de corrélation :")
        corr_matrix = df.corr()
        st.write(corr_matrix)
        st.write("""
        **Explication** : La matrice de corrélation montre les corrélations entre toutes les colonnes numériques dans le jeu de données. 
        Les coefficients de corrélation vont de -1 à +1, où +1 indique une forte corrélation positive, -1 indique une forte corrélation négative, et 0 indique aucune corrélation.
        Cela aide à identifier les relations entre les variables, ce qui est crucial pour la sélection des caractéristiques et la compréhension des dépendances dans les données.
        """)

        # Exemple : Analyse de Régression
        st.subheader("Analyse de Régression (Exemple : Régression Linéaire)")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            target_col = st.selectbox("Sélectionner la variable cible pour la régression", numeric_cols, index=0)
            feature_cols = st.multiselect("Sélectionner les variables explicatives pour la régression", numeric_cols)
            if st.button("Exécuter la Régression"):
                X = df[feature_cols]
                y = df[target_col]

                # Régression des moindres carrés ordinaires (OLS) en utilisant statsmodels
                X = sm.add_constant(X)  # Ajout d'une constante pour l'interception
                model = sm.OLS(y, X)
                results = model.fit()
                st.write(results.summary())

                # Nuage de points des valeurs réelles par rapport aux valeurs prédites
                y_pred = results.predict(X)
                fig, ax = plt.subplots()
                ax.scatter(y, y_pred)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                ax.set_xlabel('Réel')
                ax.set_ylabel('Prédit')
                ax.set_title('Réel vs Prédit')
                st.pyplot(fig)
                st.write("""
                **Explication** : La régression linéaire modèle la relation entre une variable dépendante (cible) et une ou plusieurs variables indépendantes (caractéristiques).
                Le résumé de la régression fournit des informations sur la signification et les coefficients de chaque caractéristique, aidant à interpréter leur impact sur la variable cible.
                Le nuage de points visualise à quel point les prédictions du modèle correspondent aux valeurs réelles, évaluant ainsi la performance du modèle.
                """)

                if results.rsquared > 0.5:
                    st.write("Le modèle explique plus de 50% de la variance dans la variable cible.")
                else:
                    st.write("Le modèle explique moins de 50% de la variance dans la variable cible.")
                st.write("""
                **Interprétation** : La valeur de R² indique à quel point le modèle de régression s'ajuste aux données.
                Une valeur de R² plus proche de 1 suggère que le modèle explique une plus grande proportion de la variance dans la variable cible, indiquant un meilleur ajustement.
                """)

        else:
            st.write(
                "Nombre insuffisant de colonnes numériques disponibles pour l'analyse de régression. Il faut au moins une variable cible et une variable explicative.")
            st.write("""
            **Explication** : L'analyse de régression nécessite des colonnes numériques à la fois pour la variable cible (variable dépendante) et au moins une variable explicative (variable indépendante).
            Veuillez vous assurer d'avoir suffisamment de colonnes numériques sélectionnées pour l'analyse de régression.
            """)

    else:
        st.write("Aucune donnée disponible. Veuillez télécharger ou vous connecter à une source de données sur la page de connexion aux données.")


if __name__ == "__main__":
    main()
