import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Récupération des données
def fetch_data():
    if 'final_dataframe' in st.session_state:
        return st.session_state['final_dataframe']
    else:
        st.warning("Aucun dataframe trouvé dans l'état de session.")
        return None

# Prétraitement des données
def preprocess_for_clustering(df, features):
    df_clustering = df[features].copy()
    df_encoded = pd.get_dummies(df_clustering)
    return df_encoded

# Méthode du coude pour déterminer le nombre optimal de clusters
def elbow_method(df, features):
    st.write("#### Méthode du Coude pour Déterminer le Nombre Optimal de Clusters")
    X = preprocess_for_clustering(df, features)
    sse = {}
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=42)
        try:
            kmeans.fit(X)
            sse[k] = kmeans.inertia_
        except ValueError as e:
            st.error(f"Erreur lors de l'entraînement du modèle KMeans avec {k} clusters: {e}")
            return
    fig, ax = plt.subplots()
    ax.plot(list(sse.keys()), list(sse.values()), marker='o')
    ax.set_xlabel("Nombre de clusters")
    ax.set_ylabel("SSE")
    ax.set_title("Méthode du Coude", fontweight='bold')
    st.pyplot(fig)

# Entraînement du modèle KMeans et évaluation
def kmeans_clustering(df, features, n_clusters):
    X = preprocess_for_clustering(df, features)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['kmeans_model'] = model

    st.write(f"**Nombre de clusters sélectionné : {n_clusters}**")
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    cluster_explanations = []
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Cluster {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Cluster {cluster}** : "
        explanation += f"Ce cluster contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Visualisation des Clusters avec PCA (KMeans)")
    st.pyplot(fig)

    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    st.success("Modèle KMeans entraîné et enregistré avec succès!")

# Entraînement du modèle Agglomerative Clustering et évaluation
def agglomerative_clustering(df, features, n_clusters):
    X = preprocess_for_clustering(df, features)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['agglomerative_model'] = model

    st.write(f"**Nombre de clusters sélectionné : {n_clusters}**")
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    cluster_explanations = []
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Cluster {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Cluster {cluster}** : "
        explanation += f"Ce cluster contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Visualisation des Clusters avec PCA (Agglomerative Clustering)")
    st.pyplot(fig)

# Entraînement du modèle DBSCAN et évaluation
def dbscan_clustering(df, features, eps, min_samples):
    X = preprocess_for_clustering(df, features)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['dbscan_model'] = model

    st.write(f"**Paramètres sélectionnés : eps={eps}, min_samples={min_samples}**")
    n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
    st.write(f"Nombre de clusters trouvés : {n_clusters}")

    if n_clusters < 2:
        st.warning("Le nombre de clusters trouvés est inférieur à 2, le score de silhouette ne peut pas être calculé.")
        return

    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    cluster_explanations = []
    for cluster in set(df['Cluster']):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Cluster {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Cluster {cluster}** : "
        explanation += f"Ce cluster contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Visualisation des Clusters avec PCA (DBSCAN)")
    st.pyplot(fig)

# Entraînement du modèle GMM et évaluation
def gmm_clustering(df, features, n_clusters):
    X = preprocess_for_clustering(df, features)
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['gmm_model'] = model

    st.write(f"**Nombre de clusters sélectionné : {n_clusters}**")
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    cluster_explanations = []
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Cluster {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Cluster {cluster}** : "
        explanation += f"Ce cluster contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Visualisation des Clusters avec PCA (GMM)")
    st.pyplot(fig)

# Fonction principale
def main():
    st.title("Entraînement des Modèles de Machine Learning et Clustering")

    df = fetch_data()
    if df is not None:
        st.header("Tableau de Données Utilisé")
        st.dataframe(df)

        st.header("Modélisation et Évaluation des Clusters")
        clustering_features = st.multiselect("Sélectionner les caractéristiques pour le clustering",
                                             df.columns.tolist())

        if st.button("Appliquer la Méthode du Coude"):
            if len(clustering_features) > 1:
                elbow_method(df, clustering_features)
            else:
                st.warning("Veuillez sélectionner au moins deux caractéristiques pour la méthode du coude.")

        n_clusters = st.slider("Nombre de clusters", min_value=2, max_value=20, value=6)
        if st.button("Entraîner KMeans"):
            if len(clustering_features) > 1:
                kmeans_clustering(df, clustering_features, n_clusters)
            else:
                st.warning("Veuillez sélectionner au moins deux caractéristiques pour entraîner KMeans.")

        if st.button("Entraîner Agglomerative Clustering"):
            if len(clustering_features) > 1:
                agglomerative_clustering(df, clustering_features, n_clusters)
            else:
                st.warning(
                    "Veuillez sélectionner au moins deux caractéristiques pour entraîner Agglomerative Clustering.")

        eps = st.slider("Valeur de eps pour DBSCAN", min_value=0.1, max_value=10.0, value=0.5)
        min_samples = st.slider("Nombre minimal d'échantillons pour DBSCAN", min_value=1, max_value=20, value=5)
        if st.button("Entraîner DBSCAN"):
            if len(clustering_features) > 1:
                dbscan_clustering(df, clustering_features, eps, min_samples)
            else:
                st.warning("Veuillez sélectionner au moins deux caractéristiques pour entraîner DBSCAN.")

        if st.button("Entraîner GMM"):
            if len(clustering_features) > 1:
                gmm_clustering(df, clustering_features, n_clusters)
            else:
                st.warning("Veuillez sélectionner au moins deux caractéristiques pour entraîner GMM.")

if __name__ == "__main__":
    main()
