import streamlit as st


def main():
    """
    Fonction principale pour afficher une application Streamlit avec un style personnalisé et du contenu.

    Cette fonction affiche un logo, une bannière, un titre principal, un sous-titre, et une section décrivant
    les fonctionnalités de l'application en utilisant des icônes et du balisage HTML.

    Utilisation :
        - Assurez-vous que 'assets/images/banner.jpg' existe dans votre structure de projet.
        - Exécutez l'application Streamlit avec streamlit run my_app.py.

    Exemple :
        >>> main()
    """
    # Affichage du logo
    st.logo("assets/images/logo.png", icon_image="assets/images/logo.png")

    # Affichage de la bannière
    st.image("assets/images/banner.jpg", use_column_width=True)


    # Titre de la page principale
    st.image("assets/images/logo.png", width=150)
    st.markdown('<h1 class="title">DATA DIGITAL ONE</h1>', unsafe_allow_html=True)

    # Sous-titre
    st.markdown('<h2 class="subtitle">Together, we combine Data and Digital</div>', unsafe_allow_html=True)

    st.write("""
        La segmentation des clients est une tâche fondamentale en marketing et en gestion de la relation client. 
        Avec les avancées en analyse de données et en apprentissage automatique, il est maintenant possible de 
        regrouper les clients en segments distincts avec une grande précision, permettant aux entreprises d'adapter 
        leurs stratégies marketing et leurs offres aux besoins et préférences uniques de chaque segment.

        **Problème/Requis** : Utiliser des techniques d'apprentissage automatique et d'analyse de données en Python pour effectuer la segmentation des clients.
        """)

    # Section sur l'utilisation de l'application avec des icônes
    st.markdown("""
        <div class="instruction">
            <p>Cette application propose des fonctionnalités pour :</p>
            <ul>
                <li><span class="icon">🔗</span>Connexion aux Données : Téléchargez un fichier CSV, Excel ou connectez-vous à une base de données.</li>
                <li><span class="icon">📝</span>Description des Données : Obtenez un résumé et une vue d'ensemble de vos de données.</li>
                <li><span class="icon">📈</span>Analyse des Données : Effectuez une analyse exploratoire des données.</li>
                <li><span class="icon">🔄</span>Transformation des Données : Appliquez des transformations à vos données.</li>
                <li><span class="icon">📊</span>Évaluation du Modèle : Évaluez des performances de modèles.</li>
                <li><span class="icon">🏋️</span>Entraînement du Modèle : Entraînez vos modèles d'apprentissage automatique.</li>
                <li><span class="icon">📊</span>Modélisation et Évaluation : Déterminer le nombre optimal de clusters avec l'analyse de la silhouette.</li>
                <li><span class="icon">📉</span>Visualisation des Données : Visualisez les résultats de vos données.</li>
                <li><span class="icon">💡</span>Suggestions de Nettoyage des Données : Obtenez des suggestions pour nettoyer vos données.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Fermer les conteneurs principaux
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
