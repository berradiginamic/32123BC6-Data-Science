# In traitement/normal_distribution.py
from scipy.stats import norm, shapiro, expon
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


def distribution_pairplot(df, selected_columns, target_column):
    if st.sidebar.checkbox("Graphique de distribution et pairplot"):
        st.write("Graphique de distribution :")
        if target_column:
            st.pyplot(sns.pairplot(df[selected_columns + [target_column]]).fig)
        else:
            st.write("S'il vous plait selectionnez une colonne cible pour afficher les graphiques")



def visualize_normal_distribution(df, selected_columns):
    if st.sidebar.checkbox("Distribution Normale"):
        mean = st.slider("Moyenne", float(df[selected_columns].mean()), float(df[selected_columns].mean() + 10.0))
        std_dev = st.slider("Écart-type", float(df[selected_columns].std()), float(df[selected_columns].std() + 5.0))

        x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 1000)
        y = norm.pdf(x, mean, std_dev)

        plt.plot(x, y)
        st.pyplot(plt)

        # Test de normalité pour la colonne sélectionnée
        stat, p_value = shapiro(df[selected_columns])

        # Afficher le résultat du test
        st.write(f"Test de Normalité (Shapiro-Wilk) pour la colonne '{selected_columns}':")
        st.write(f"Statistique de test : {stat}, P-valeur : {p_value}")

        # Interprétation du test
        alpha = 0.05  # Niveau de signification
        if p_value > alpha:
            st.write("Les données semblent suivre une distribution normale.")
        else:
            st.write("Les données ne suivent pas une distribution normale.")

        # Formule mathématique et explication
        st.latex(r'''
        f(x|\mu,\sigma^2) = \frac{1}{\sigma \sqrt{2\pi}} e^{ -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2 }
        ''')
        st.write("où:")
        st.write(r"$\mu$ : Moyenne")
        st.write(r"$\sigma$ : Écart-type")


def visualize_exponential_distribution(df, selected_columns):
    if st.sidebar.checkbox("Distribution Exponentielle"):
        scale_param = st.slider("Paramètre d'échelle", float(df[selected_columns].mean()),
                                float(df[selected_columns].mean() + 5.0))

        x = np.linspace(0, 10, 1000)
        y = expon.pdf(x, scale=scale_param)

        plt.plot(x, y)
        st.pyplot(plt)

        # Formule mathématique et explication
        st.latex(r'''
        f(x|\lambda) = \lambda e^{-\lambda x}
        ''')
        st.write("où:")
        st.write(r"$\lambda$ : Paramètre d'échelle")