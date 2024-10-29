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

st.title('Clusteranalyse')

# st.header('asdf')

# Define the list of acceptable model types
supported_methods = {"k-Means": "kMeans",
                     "DBSCAM": "DBSCAN"
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

st.write("""Lade die Trainingsdaten als csv-Datei in folgendem Format hoch:\\
             Erste Zeile: Spaltenbezeichnungen. 
             Eine Spalte pro Feature (unabhängige Variablen).
             Alle Einträge müssen in numerischer Form kodiert vorliegen. Beispiel mit 3 Features und 4 Samples (mit der Kopzeile insgesamt 5 Zeilen):""")
df = pd.DataFrame({'Feature 1': [0.0, 4.1, 2.3, 1.9], 'Feature 2': [0, 1, 1, 0], 
                              'Feature 3': [5, 3, 9, 2]}
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

    # Extract columns
    columns = df.columns.tolist()
        
    # User selects predictors and target variable
    predictors = st.multiselect("Wähle die Merkmale, die im Training berücksichtig werden sollen", columns)

    if predictors:
        # Slice for selected predictors and target
        X = df[predictors]

        # Consistency check: no missing values, only numeric values allowed
        if X.isnull().values.any():
            st.error("Die Daten enthalten fehlende Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
        elif not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            st.error("Die Daten enthalten nicht-numerische Werte. Bitte bereinige die Daten und lade sie erneut hoch.")
        else:

            # User selects dimensionality Clustering method
            selected_method = st.selectbox("Wähle die Methode", ["k-Means", "DBSCAN"])

            if selected_method == "k-Means":
                from sklearn.cluster import KMeans
                n_clusters = st.number_input("Wähle die Anzahl der Cluster", min_value=1, value=3)
                init = st.selectbox("Wähle die Initialisierungsmethode", ["k-means++", "random"])
                max_iter = st.number_input("Wähle die maximale Anzahl der Iterationen", min_value=100, value=300)
                model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter)
                model.hyperpars_str = f'n_clusters: {n_clusters}, init: {init}, max_iter: {max_iter}'
            elif selected_method == "DBSCAN":
                from sklearn.cluster import DBSCAN
                eps = st.number_input("Wähle den Radius der Nachbarschaft (eps)", min_value=0.1, value=0.5, step=0.1)
                min_samples = st.number_input("Wähle die minimale Anzahl an Samples in Nachbarschaft", min_value=1, value=5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                model.hyperpars_str = f'eps: {eps}, min_samples: {min_samples}'
                
            model.method = supported_methods[selected_method]
            
            # Train the  model
            if st.button("Modell trainieren"):
                model.id = str(uuid.uuid4())  # Generate a unique alphanumeric ID
                model.fit(X)
                model.date_trained = datetime.now()
                st.success("Das Modell wurde erfolgreich trainiert.")

                # Save the model to session state
                st.session_state['trained_model'] = model
                st.session_state['predictors'] = predictors
                st.session_state['X'] = X
                st.session_state['labels'] = model.labels_

            if 'trained_model' in st.session_state and st.session_state['trained_model']:
                # Load the trained model from session state
                model = st.session_state['trained_model']
                predictors = st.session_state['predictors']
                labels = st.session_state['labels']
                # Allow user to download the serialized model
                model_file = io.BytesIO()
                joblib.dump(model, model_file)
                model_file.seek(0)
                
                st.download_button("Trainiertes Modell herunterladen", model_file,
                                file_name=f"{model.method}_{model.date_trained.strftime('%M-%H-%d-%m-%Y')}_{model.id}.pkl")

                # Create a DataFrame for the features and derived labels
                df_features_labels = pd.DataFrame(X, columns=predictors)
                df_features_labels['Cluster'] = labels

                df_modelinfo = pd.DataFrame([[get_key_from_value(supported_methods, model.method), model.hyperpars_str, model.id,
                                    model.date_trained.strftime("%d.%m.%Y, %H:%M"), '|'.join(predictors)]],
                                    columns=['Methode', 'Hyperparameter', 'Modell-Id', 'Trainingszeitpunkt', 'Merkmale']
                                    )

                # Display predictions
                st.write(f"Samples mit zugeordneten Clustern:")
                st.dataframe(df_features_labels, hide_index=True)

                # Create a Pandas Excel writer using openpyxl as the engine
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Write df_features_labels to the first sheet
                    df_features_labels.to_excel(writer, sheet_name='Merkmale und Labels', index=False)

                    # Write df_modelinfo to the second sheet
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
                st.download_button("Transformierte Daten herunterladen", excel_buffer, 
                                file_name=f"Inferenz_{model.date_trained.strftime('%M-%H-%d-%m-%Y')}_{model.id}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                
                st.subheader('Evaluation')
            