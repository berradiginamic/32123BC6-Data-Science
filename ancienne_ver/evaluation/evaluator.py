from turtle import st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Function to evaluate the trained model
def evaluate_model(model, selected_model, X_test, y_test):
    if selected_model == "Dimensionality Reduction Algorithms":
        st.warning("Impossible d'évaluer un modèle de réduction de dimension (PCA) de cette manière.")
    else:
        if not isinstance(model, PCA):
            if selected_model == "Linear Regression":
                st.warning("La régression linéaire n'est pas adaptée à la classification. Choisissez un modèle de classification approprié.")
            else:
                y_pred = model.predict(X_test)

                if selected_model == "Logistic Regression":
                    # Convert predictions into classes (0 or 1) for binary classification
                    y_pred = np.round(y_pred)

                accuracy = accuracy_score(y_test, y_pred)
                st.write("Précision du modèle :", accuracy)
                st.write("valeurs prédites :", model.predict(X_test))
                st.write("vrai valeurs:", y_test)

                # Confusion matrix
                st.write("Matrice de confusion :")
                confusion_matrix_display = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
                st.pyplot(confusion_matrix_display.figure)

                # Classification report
                st.write("Rapport de classification :")
                classification_rep = classification_report(y_test, y_pred, output_dict=True)

                # Convert classification report to DataFrame for better formatting
                df_classification_rep = pd.DataFrame(classification_rep).transpose()
                st.write(df_classification_rep)

                # Basic interpretation of the classification report
                st.write("Interprétation du rapport de classification :")
                st.write("L'accuracy du modèle est :", classification_rep['accuracy'])

                # Provide insights into precision, recall, and F1-score
                for class_label, metrics in classification_rep.items():
                    if class_label != 'accuracy':
                        st.write(f"Classe {class_label}:")
                        st.write(f"Precision: {metrics['precision']}")
                        st.write(f"Recall: {metrics['recall']}")
                        st.write(f"F1-score: {metrics['f1-score']}")
                        st.write("\n")
        else:
            st.warning("Impossible d'évaluer un modèle de réduction de dimension (PCA) de cette manière.")

