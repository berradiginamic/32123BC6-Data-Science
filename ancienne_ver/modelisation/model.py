# In traitement/model_training.py
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit as st


# Function to train the selected machine learning model

# Function to train machine learning model
def train_machine_learning_model(selected_model, X_train, y_train, num_epochs):
    model = None
    if selected_model == "Linear Regression":
        model = LinearRegression()
    elif selected_model == "Logistic Regression":
        model = LogisticRegression()
    elif selected_model == "Decision Tree":
        model = DecisionTreeClassifier()
    elif selected_model == "SVM":
        model = SVC()
    elif selected_model == "Naive Bayes":
        model = GaussianNB()
    elif selected_model == "Random Forest":
        model = RandomForestClassifier()
    elif selected_model == "Dimensionality Reduction Algorithms":
        n_components = min(X_train.shape[0], X_train.shape[1])
        model = PCA(n_components=n_components)
    else:
        st.warning("Modèle non pris en charge : {}".format(selected_model))

    if model is not None:
        for epoch in range(num_epochs):
            model.fit(X_train, y_train)
            # You can add progress indicators or logging here if needed
        st.success("Le modèle a été entraîné avec succès sur {} epochs!".format(num_epochs))
    return model


# Function to get user input for making predictions
def get_user_input(selected_columns):
    user_input = {}
    for column in selected_columns:
        value = st.text_input(f"Entrez la valeur pour {column}:")
        user_input[column] = value
    return pd.DataFrame([user_input])
