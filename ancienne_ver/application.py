import pandas as pd
import streamlit as st
from reportlab.pdfgen import canvas
from ancienne_ver.utils.lazyutils import LazyClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ancienne_ver.evaluation.evaluator import evaluate_model
from ancienne_ver.modelisation.model import get_user_input, train_machine_learning_model
from ancienne_ver.traitement.description import *
from ancienne_ver.traitement import *
from ancienne_ver.traitement.nettoyage import *

# ----------------------------------------------------------------------------------------------------------------------
# Étape 1: Construction de Streamlit
st.sidebar.title("Paramètres")

st.title("Application Machine Learning")

# ----------------------------------------------------------------------------------------------------------------------
# Étape 2: Chargement du jeu de données
# df = load_data()
st.session_state.df = load_data()
df = st.session_state.df
if df is not None:
    if st.checkbox('Détecter les valeurs manquantes'):
        detect_na(df)

    if st.checkbox('Détecter les outliers'):
        outliers = detect_outliers(df)
        if outliers.empty:
            st.write("Aucun outlier détecté.")
        else:
            st.write("Outliers détectés :")
            st.write(outliers)

    if st.checkbox('Traiter les outliers'):
        threshold = st.slider("Choisir le seuil de Z-score :", min_value=1, max_value=10, value=3, step=1)
        df = treat_outliers(df, threshold)
        st.write("Après le traitement des outliers :")
        st.write(df.head())

    if st.checkbox('Supprimer les colonnes vides'):
        df = drop_empty_columns(df)

    if st.checkbox('Detecter et supprimer les colonnes similaires à l\'index'):
        df = detect_and_drop_similar_to_index_columns(df)

    if st.checkbox('Afficher les valeurs uniques dans chaque colonne'):
        count_unique_values(df)

    df = handle_missing_values(df)

    st.write("Après suppression des colonnes vides et similaires à l'index :")
    st.write(df.head())

    selected_columns, target_column = show_data_processing_options(df)

    st.session_state.df = df

# ----------------------------------------------------------------------------------------------------------------------
# Étape 3: Traitement des données

st.sidebar.write("Traitements variables:")

# Analyse descriptive
descriptive_analysis(df[selected_columns])

# Graphique de distribution et pairplot
distribution_pairplot(df, df[selected_columns], df[target_column])

# Corrélation avec la cible
correlation_with_target(df[selected_columns], df[target_column])

# Ajouté ici code standardisation
standardization(df[selected_columns])
# Ajouté ici code encodage

st.sidebar.write("Traitements variable cible:")

# Fréquences
column_frequencies(df, target_column)

# Visualisation de la distribution normale
visualize_normal_distribution(df, target_column)

# Visualisation de la distribution exponentielle
visualize_exponential_distribution(df, target_column)
# --------------------------------------------------------------------------
# Étape 4: Machine Learning
# Sidebar Section: Model Training
st.sidebar.subheader("Bloc de Machine Learning")

# Initialization of the model to None
#model = None
# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        'model': None
    }
train_model = st.sidebar.toggle("Entraîner le modèle")

if train_model:
    selected_model = st.sidebar.selectbox("Sélectionnez le modèle", [" ", "Linear Regression", "Logistic Regression",
                                                                     "Decision Tree", "SVM", "Naive Bayes",
                                                                     "Random Forest",
                                                                     "Dimensionality Reduction Algorithms"])
    # Sidebar Section: Model-specific parameters
    st.sidebar.subheader("Paramètres spécifiques au modèle")

    if selected_model == "Linear Regression":
        #num_epochs = st.sidebar.slider("Nombre d'époques", min_value=1, max_value=100, value=10)
        # Ajoutez d'autres paramètres spécifiques si nécessaire
        pass

    elif selected_model == "Logistic Regression":
        # Ajoutez les paramètres spécifiques au modèle Logistic Regression
        C = st.sidebar.slider("Paramètre C", min_value=0.01, max_value=10.0, value=1.0)
        pass

    elif selected_model == "Decision Tree":
        # Ajoutez les paramètres spécifiques au modèle Decision Tree
        max_depth = st.sidebar.slider("Profondeur maximale de l'arbre", min_value=1, max_value=20, value=5)
        pass

    elif selected_model == "SVM":
        # Ajoutez les paramètres spécifiques au modèle SVM
        C = st.sidebar.slider("Paramètre C", min_value=0.01, max_value=10.0, value=1.0)
        pass

    elif selected_model == "Naive Bayes":
        # Ajoutez les paramètres spécifiques au modèle Naive Bayes
        # Aucun paramètre spécifique pour le modèle Naive Bayes
        pass

    elif selected_model == "Random Forest":
        # Ajoutez les paramètres spécifiques au modèle Random Forest
        n_estimators = st.sidebar.slider("Nombre d'arbres", min_value=1, max_value=100, value=10)
        max_depth = st.sidebar.slider("Profondeur maximale des arbres", min_value=1, max_value=20, value=5)
        pass

    elif selected_model == "Dimensionality Reduction Algorithms":
        # Utilisez le nombre de composants principaux comme valeur par défaut pour n_components
        n_components = st.sidebar.slider("Nombre de composants principaux", min_value=0, max_value=df.shape[1],
                                         value=15)
        if n_components > min(X_train.shape[0], X_train.shape[1]):
            st.warning("La valeur de n_components est trop élevée. Elle a été ajustée à la valeur maximale possible.")
            n_components = min(X_train.shape[0], X_train.shape[1])
        # Assurez-vous que n_components est au moins égal à 1
        n_components = max(n_components, 1)
        pass

    num_epochs = st.sidebar.number_input("Nombre d'époques", min_value=1, value=10)

    if selected_model and selected_columns and target_column:
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2,
                                                            random_state=42)
        # Button to trigger model training
        if st.sidebar.button("Lancer l'entraînement"):
            # Model Training
            st.session_state.state['model'] = train_machine_learning_model(selected_model, X_train, y_train, num_epochs)

        # from sklearn.preprocessing import LabelEncoder
        # label_encoder = LabelEncoder()
        # y_train = label_encoder.fit_transform(y_train)
        # y_test = label_encoder.transform(y_test)

    # Model Training
    #model = train_machine_learning_model(selected_model, X_train, y_train, num_epochs)

# -----------------------------------------------------------------------
# Initialisation du tableau pour stocker les métriques
metrics_table = pd.DataFrame(columns=["Modèle", "MSE", "MAE"])

def save_as_pdf(dataframe, filename):
    # Créer un fichier PDF avec ReportLab
    c = canvas.Canvas(filename)

    # Titre du PDF
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 800, "Rapport de Résultats")

    # Contenu du PDF (utilisez les données du DataFrame)
    c.setFont("Helvetica", 12)

    # Ajoutez chaque ligne de votre DataFrame séparément
    y_position = 750
    for line in dataframe.to_string(index=False).split('\n'):
        c.drawString(100, y_position, line)
        y_position -= 12  # Ajustez selon la taille de la police et l'espacement souhaité

    # Enregistrez le PDF
    c.save()
    st.success(f"Le fichier PDF '{filename}' a été créé avec succès!")

# Comparaison des modèles
if st.sidebar.toggle("Comparer les modèles"):
    st.subheader("Comparaison des modèles:")

    # Initialisation des modèles
    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        # "Dimensionality Reduction Algorithms": PCA(n_components=n_components)  # Utilisez la valeur spécifiée
    }

    for model_name, model in models.items():
        st.write(f"Modèle : {model_name}")

        try:
            if model_name:  # != "Dimensionality Reduction Algorithms":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train)
                y_pred = model.transform(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Stocker les métriques dans le tableau
            metrics_table = pd.concat([metrics_table, pd.DataFrame({"Modèle": [model_name], "MSE": [mse], "MAE": [mae]})],
                                      ignore_index=True)

        except Exception as e:
            st.warning(f"Erreur lors de l'exécution du modèle {model_name}. Message d'erreur : {str(e)}")

    # Enregistrer le tableau de comparaison en PDF
    # Enregistrer le message de comparaison en pdf


# Afficher le tableau de comparaison
st.write("Tableau de comparaison des modèles:")
st.write(metrics_table)

# Trier le tableau par MSE croissante
if not metrics_table.empty:
    sorted_table_mse = metrics_table.sort_values(by="MSE")

    # Vérifier si le DataFrame n'est pas vide
    if not sorted_table_mse.empty:
        # Afficher le modèle avec la plus faible MSE
        best_model_mse = sorted_table_mse.iloc[0]["Modèle"]
        st.write(f"Meilleur modèle selon la MSE : {best_model_mse}")
    else:
        st.write("Le DataFrame est vide. Aucun meilleur modèle trouvé.")
else:
    st.write("Le DataFrame est vide. Aucun modèle n'a été entraîné.")

# Trier le tableau par MAE croissante
if not metrics_table.empty:
    sorted_table_mae = metrics_table.sort_values(by="MAE")

    # Vérifier si le DataFrame n'est pas vide
    if not sorted_table_mae.empty:
        # Afficher le modèle avec la plus faible MAE
        best_model_mae = sorted_table_mae.iloc[0]["Modèle"]
        st.write(f"Meilleur modèle selon la MAE : {best_model_mae}")
    else:
        st.write("Le DataFrame est vide. Aucun meilleur modèle trouvé.")
else:
    st.write("Le DataFrame est vide. Aucun modèle n'a été entraîné.")

 # Enregistrer le message de comparaison en pdf
if st.button("Télécharger le rapport en PDF", key="ab"):
    save_as_pdf(metrics_table, "rapport_resultats.pdf")
# -------------------------------------------------------------------------------------------------------
# Étape 5: Evaluation Modele
# Sidebar Section: Model Evaluation
st.sidebar.subheader("Bloc d'Évaluation")

# Model Evaluation
if st.sidebar.button("Évaluer le modèle"):
    if st.session_state.state['model'] is not None:
        evaluate_model(st.session_state.state['model'], selected_model, X_test, y_test)




# Sidebar Section: Predictions on New Data
if st.sidebar.checkbox("Prédictions sur de nouvelles données"):
    st.write("Entrez les nouvelles données à prédire :")
    new_data = get_user_input(selected_columns)

    if model is not None and not isinstance(model, PCA):
        prediction = model.predict(new_data)
        st.write("Prédiction :", prediction)
    elif isinstance(model, PCA):
        st.warning("Impossible de faire des prédictions avec un modèle de réduction de dimension (PCA).")
    else:
        st.warning("Aucun modèle n'est sélectionné.")
# -------------------------------------------------------------------------------------------------------
# Étape 6: Fonctionnalités supplémentaires
st.sidebar.subheader("Fonctionnalités Supplémentaires")

# Définition de la fonction Lazy Predict
def lazy_predict(X_train, X_test, y_train, y_test):
    # Utiliser LazyClassifier pour prédire automatiquement avec plusieurs modèles
    clf = LazyClassifier()
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Afficher ou retourner les résultats
    st.write(models)
    #st.write(predictions)

# Définition de la fonction GridSearchCV
def grid_search_cv(X_train, y_train):
    # Créer un modèle RandomForestClassifier (exemple)
    model = RandomForestClassifier()

    # Définir la grille des hyperparamètres à rechercher
    param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}

    # Appliquer la recherche par grille
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs hyperparamètres trouvés
    st.write("Meilleurs hyperparamètres :", grid_search.best_params_)

# Définition de la fonction Keras Model
def keras_model(X_train, X_test, y_train, y_test):
    # Créer un modèle séquentiel simple (exemple)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Évaluer le modèle
    accuracy = model.evaluate(X_test, y_test)[1]
    st.write("Précision du modèle :", accuracy)

# Utilisation des fonctions dans le code principal
#st.sidebar.subheader("Fonctionnalités Supplémentaires")

# Lazy Predict
if st.sidebar.checkbox("Lazy Predict"):
    st.write("Lazy Predict :")
    lazy_predict(X_train, X_test, y_train, y_test)

# GridSearchCV
if st.sidebar.checkbox("GridSearchCV"):
    st.write("GridSearchCV :")
    grid_search_cv(X_train, y_train)

# Modèle de Deep Learning (Exemple avec Keras)
if st.sidebar.checkbox("Modèle de Deep Learning (Keras)"):
    st.write("Modèle de Deep Learning avec Keras :")
    keras_model(X_train, X_test, y_train, y_test)