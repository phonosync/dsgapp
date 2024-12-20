import streamlit as st
import streamlit as st
import pandas as pd
import openpyxl
from openpyxl.styles import NamedStyle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
import joblib
import io
import uuid
from datetime import datetime
import utils

st.title('Klassifikationsmethoden')

st.header('Training und Evaluation eines Klassifikationsmodells')

# Define the list of acceptable model types
supported_methods = {"Nächste Nachbarn-Heuristik": "KNeighborsClassification",
                     "Logistische Regression": "LogisticRegression",
                     "Support Vector Machines": "SupportVectorMachines",
                     "Dummy - Zufall": "DummyRandom",
                     "Dummy - Häufigste Klasse": "DummyMostFrequent"
                    }

def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

@st.cache_data
def load_data(uploaded_file):
    return utils.inp_file_2_csv(inp_file=uploaded_file, index_column=False,
                                  header_row=True)

st.subheader('Training')
st.write('Trainiere ein Klassifikationsmodell auf den Trainingsdaten in einer CSV-Datei')
st.write("""Lade die Trainingsdaten als csv-Datei in folgendem Format hoch:\\
             Erste Zeile: Spaltenbezeichnungen. 
             Eine Spalte pro Feature (unabhängige Variablen) und für die
              Zielvariable (Labels, die es mit dem zu trainierenden Modell vorherzusagen gilt).
             Alle Einträge müssen in numerischer Form kodiert vorliegen. Beispiel mit 3 Features, 
             der Zielvariable (insgesamt 4 Spalten) und 4 Samples (mit der Kopzeile insgesamt 5 Zeilen):""")
df = pd.DataFrame({'Feature 1': [0.0, 4.1, 2.3, 1.9], 'Feature 2': [0, 1, 1, 0], 
                              'Feature 3': [5, 3, 9, 2], 'Target': [0, 1, 1, 1]}
                              )

st.dataframe(df, hide_index=True)

# File uploader for training data
train_file = st.file_uploader("Wähle eine CSV-Datei mit den Trainingsdaten", type="csv")

if train_file is not None:
    # Read the CSV file
    df = load_data(train_file)
    
    # Display the DataFrame
    st.write("Auszug aus den hochgeladenen Trainingsdaten:")
    st.dataframe(df[:5], hide_index=True)
    
    # Consistency check: no missing values, only numeric values allowed
    if df.isnull().values.any():
        st.error("Die Daten enthalten fehlende Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
    elif not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        st.error("Die Daten enthalten nicht-numerische Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
    else:
        # Extract columns
        columns = df.columns.tolist()
        
        # User selects predictors and target variable
        predictors = st.multiselect("Wähle die unabhängigen Merkmale, die im Training berücksichtig werden sollen", columns)
        target = st.selectbox("Wähle die abhängige Variable (Zielvariable/Labels)", columns[::-1])

        if predictors and target:
            # Slice for selected predictors and target
            X = df[predictors]
            y = df[target]

            # User selects classification algorithm
            selected_method = st.selectbox("Wähle den Klassifikationsalgorithmus", [m for m in supported_methods.keys()])
            
            if selected_method == "Logistische Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()
                model.hyperpars_str = ''
            elif selected_method == "Support Vector Machines":
                from sklearn.svm import SVC
                C = st.number_input("Wähle den Parameter C", min_value=0.01, value=1.0)
                kernel = st.selectbox("Wähle den Kernel", ["linear", "poly", "rbf", "sigmoid"])
                degree = st.number_input("Wähle den Grad des Polynoms", min_value=1, value=3, step=1) if kernel == "poly" else 3
                gamma = st.number_input("Wähle Gamma für den 'rbf'-Kernel)", min_value=0.01, value=0.1) if kernel == "rbf" else 1.0

                model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
                model.hyperpars_str = f'C: {C}, Kernel: {kernel}, Grad: {degree}, Gamma: {gamma}'
            elif selected_method == "Nächste Nachbarn-Heuristik":
                from sklearn.neighbors import KNeighborsClassifier
                n_neighbors = st.number_input("Wähle die Anzahl der Nachbarn (k)", min_value=1, value=3)
                metric = st.selectbox("Wähle die Ähnlichkeitsmetrik", ["manhattan", "euclidean", "cosine"])
                model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                model.hyperpars_str = f'k: {n_neighbors}, Distanz-Metrik: {metric}'
            elif selected_method == "Dummy - Zufall":
                from sklearn.dummy import DummyClassifier
                model = DummyClassifier(strategy="uniform")
                model.hyperpars_str = ''
            elif selected_method == "Dummy - Häufigste Klasse":
                from sklearn.dummy import DummyClassifier
                model = DummyClassifier(strategy="most_frequent")
                model.hyperpars_str = ''
            
            model.method = supported_methods[selected_method]
            
            # Train the classification model
            if st.button("Modell trainieren"):
                model.id = str(uuid.uuid4())  # Generate a unique alphanumeric ID
                model.fit(X, y)
                model.date_trained = datetime.now()
                st.success("Das Modell wurde erfolgreich trainiert.")

                # Save the model to session state
                st.session_state['trained_model'] = model
                st.session_state['predictors'] = predictors
                st.session_state['target'] = target

            if 'trained_model' in st.session_state and st.session_state['trained_model']:
                # Load the trained model from session state
                model = st.session_state['trained_model']
                predictors = st.session_state['predictors']
                target = st.session_state['target']
                # Allow user to download the serialized model
                model_file = io.BytesIO()
                joblib.dump(model, model_file)
                model_file.seek(0)
                
                st.download_button("Trainiertes Modell herunterladen", model_file,
                                file_name=f"{model.method}_{model.date_trained.strftime('%M-%H-%d-%m-%Y')}_{model.id}.pkl")

                st.subheader('Evaluation')
                st.write('Teste das trainierte Modell auf einem unabhängigen Test-Datensatz')
                # File uploader for test data
                test_file = st.file_uploader("Wähle eine CSV-Datei mit den Testdaten", type="csv")

                if test_file is not None:
                    # Read the CSV file
                    test_df = load_data(test_file)
                    
                    # Display the DataFrame
                    st.write("Auszug aus den hochgeladenen Testdaten:")
                    st.dataframe(test_df[:5], hide_index=True)
                    
                    # Consistency check: no missing values, only numeric values allowed, and all columns used in training are present
                    if test_df.isnull().values.any():
                        st.error("Die Testdaten enthalten fehlende Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
                    elif not all(test_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                        st.error("Die Testdaten enthalten nicht-numerische Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
                    elif not all(col in test_df.columns for col in predictors):
                        st.error("Die Testdaten enthalten nicht alle für das Training verwendeten Spalten. Bitte überprüfe die Daten und lade sie erneut hoch.")
                    else:
                        # Make predictions
                        X_test = test_df[predictors]
                        y_test = test_df[target]
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        conf_matrix = confusion_matrix(y_test, y_pred)
                        
                        # Display metrics
                        # Create DataFrame
                        df_metrics = pd.DataFrame([[selected_method, model.hyperpars_str, model.id, accuracy, precision, recall, f1]],
                                                columns=['Methode', 'Hyperparameter', 'Modell-Id', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
                        st.write("Evaluationsmetriken berechnet für die Vorhersagen auf den Testdaten:")
                        st.dataframe(df_metrics, hide_index=True)

                        # Display confusion matrix
                        st.write("Konfusionsmatrix:")
                        st.dataframe(pd.DataFrame(conf_matrix))

st.subheader('Inferenz')
st.write('Berechne Vorhersagen basierend auf einem bereits trainierten Modell mit neuen Eingabewerten.')

# File uploader for trained model
model_file = st.file_uploader("Wähle eine Datei mit einem trainierten Modell", type="pkl")

if model_file is not None:
    # Load the trained model
    model = joblib.load(model_file)
    
    # Check if the model is a scikit-learn model and is one of the acceptable types
    if isinstance(model, BaseEstimator) and model.method in supported_methods.values():
        # Display model type and hyperparameters

        # if hasattr(model, 'get_params'):
        #    st.write("Hyperparameter:")
        #    st.json(model.get_params())
        
        # Display dependent variables (predictors)
        if hasattr(model, 'feature_names_in_'):
            predictors = model.feature_names_in_
            # st.write("Abhängige Variablen (Prädiktoren):", predictors)
        else:
            st.error("Das Modell enthält keine Informationen über die abhängigen Variablen.")

        df_modelinfo = pd.DataFrame([[get_key_from_value(supported_methods, model.method), model.hyperpars_str, model.id,
                                      model.date_trained.strftime("%d.%m.%Y, %H:%M"), '|'.join(predictors)]],
                                    columns=['Methode', 'Hyperparameter', 'Modell-Id', 'Trainingszeitpunkt', 'Unabhängige Variablen']
                                    )
        st.write("Hochgeladenes Modell erkannt:")
        st.dataframe(df_modelinfo, hide_index=True)

        # Option for user to provide values for all input features
        st.write("Berechne eine einzelne Vorhersage basierend auf Eingabewerten:")
        input_values = {}
        for predictor in predictors:
            input_values[predictor] = st.number_input(f"Wert für {predictor}", value=0.0)

        if st.button("Vorhersage berechnen"):
            input_df = pd.DataFrame([input_values])
            prediction = model.predict(input_df)
            st.write(f"Vorhergesagter Wert: {prediction[0]:.2f}")

        st.write('Berechne Vorhersagen auf Eingabewerte in einer Datei:')
        # File uploader for new data
        new_data_file = st.file_uploader("Wähle eine CSV-Datei mit neuen Daten für Vorhersagen", type="csv")
        
        if new_data_file is not None:
            # Read the CSV file
            new_data_df = load_data(new_data_file)
            
            # Display the DataFrame
            st.write("Auszug aus den hochgeladenen Daten für Vorhersagen:")
            st.dataframe(new_data_df[:5], hide_index=True)
            
            # Consistency check: no missing values, only numeric values allowed, and all predictors are present
            if new_data_df.isnull().values.any():
                st.error("Die Daten enthalten fehlende Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
            elif not all(new_data_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                st.error("Die Daten enthalten nicht-numerische Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
            elif not all(col in new_data_df.columns for col in predictors):
                st.error("Die Daten enthalten nicht alle für das Modell erforderlichen unabhängigen Variablen. Bitte überprüfe die Daten und lade sie erneut hoch.")
            else:
                # Make predictions
                X_new = new_data_df[predictors]
            
            time_predict = datetime.now()
            predictions = model.predict(X_new)
            
            # Create a DataFrame for the predictions
            predictions_df = pd.DataFrame(predictions, columns=["Vorhersagen"])
            
            # Display predictions
            st.write(f"Vorhersagen basierend auf {', '.join(predictors)} aus den hochgeladenen Daten:")
            st.dataframe(predictions_df, hide_index=True)
            
            # Allow user to download the predictions as a CSV file

            # Create a Pandas Excel writer using openpyxl as the engine
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Write predictions_df to the first sheet
                predictions_df.to_excel(writer, sheet_name='Vorhersagen', index=False)

                # Write the input values to second sheet
                new_data_df.to_excel(writer, sheet_name='Eingabewerte', index=False)

                # Write df_modelinfo to the third sheet
                df_modelinfo.to_excel(writer, sheet_name='Modell-Infos', index=False)
                
                
            # Load the workbook to clear formatting
            excel_buffer.seek(0)
            workbook = openpyxl.load_workbook(excel_buffer)

            # Clear formatting from all cells
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    for cell in row:
                        cell.style = NamedStyle(name="normal")

            # Save the workbook to the buffer
            excel_buffer = io.BytesIO()
            workbook.save(excel_buffer)
            excel_buffer.seek(0)
            # Allow user to download the Excel file
            st.download_button("Vorhersagen herunterladen", excel_buffer, 
                               file_name=f"Inferenz_{time_predict.strftime('%M-%H-%d-%m-%Y')}_{model.id}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                               )
    else:
        st.error("Es sind nur Modelle zulässig, welche in der DSG-App trainiert wurden.")