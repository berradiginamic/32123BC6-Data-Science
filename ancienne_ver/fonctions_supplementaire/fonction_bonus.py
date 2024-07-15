# Function for Lazy Predict
import streamlit as st
import keras
from keras.models import Sequential
from keras.layers import Dense
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import GridSearchCV


def lazy_predict(X_train, X_test, y_train, y_test):
    # Determine task type (classification or regression)
    task_type = "classification" if len(set(y_train)) <= 2 else "regression"

    if task_type == "classification":
        clf = LazyClassifier(predictions=True)
        clf.fit(X_train, X_test, y_train, y_test)
        st.write(clf)
    else:
        reg = LazyRegressor(predictions=True)
        reg.fit(X_train, X_test, y_train, y_test)
        st.write(reg)


# Function for GridSearchCV
def grid_search_cv(X_train, y_train):
    # Define your grid search parameters
    parameters = {
        # Define your parameters for GridSearchCV
    }

    # Initialize your model
    model = None  # Define your model

    # Initialize GridSearchCV
    grid_search = GridSearchCV(model, parameters, cv=5, n_jobs=-1)

    # Fit the GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    st.write("Best Parameters:", best_params)
    st.write("Best Score:", best_score)


# Function for Keras model
def keras_model(X_train, X_test, y_train, y_test):
    # Define your Keras model
    model = Sequential()
    # Add layers to your model

    # Compile your model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train your model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate your model
    _, accuracy = model.evaluate(X_test, y_test)
    st.write('Accuracy: %.2f' % (accuracy * 100))
