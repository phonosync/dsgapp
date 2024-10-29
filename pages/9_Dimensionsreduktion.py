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

st.title('Dimensionsreduktion')

# st.header('asdf')

# Define the list of acceptable model types
supported_methods = {"Hauptkomponentenanalyse": "PCA",
                     "t-SNE": "tSNE"
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
            # User selects number of desired dimensions
            n_components = st.number_input("Wähle die Anzahl der zu berechnenden Dimensionen", min_value=1,
                                        max_value=X.shape[1], value=np.min([2, X.shape[1]]))

            # User selects dimensionality reduction method
            selected_method = st.selectbox("Wähle die Methode", ["Hauptkomponentenanalyse", "t-SNE"])

            if selected_method == "Hauptkomponentenanalyse":
                from sklearn.decomposition import PCA
                model = PCA(n_components=n_components)

                model.hyperpars_str = f'n_components: {n_components}'
            elif selected_method == "t-SNE":
                from sklearn.manifold import TSNE
                st.write("Zu wählende Hyperparameter:")
                perplexity = st.number_input("Wähle die Perplexität", min_value=5, value=30)
                learning_rate = st.number_input("Wähle die Lernrate", min_value=10, value=200)
                max_iter = st.number_input("Wähle die Anzahl der Iterationen", min_value=250, value=1000)
                # method = st.selectbox("Wähle die Methode", ["barnes_hut", "exact"])
                model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter)

                model.hyperpars_str = f'n_components: {n_components}, perplexity: {perplexity}, learning_rate: {learning_rate}, max_iter: {max_iter}'
                    
            model.method_label = supported_methods[selected_method]

            st.session_state['trained_model'] = None
            
            # Train the  model
            if st.button("Modell trainieren"):
                model.id = str(uuid.uuid4())  # Generate a unique alphanumeric ID
                X_transformed = model.fit_transform(X)
                model.date_trained = datetime.now()
                st.success("Das Modell wurde erfolgreich trainiert.")

                # Calculate evaluation metrics
                if model.method_label == 'PCA':
                    explained_variance = model.explained_variance_ratio_
                    # st.write("Erklärte Varianzverhältnisse der Hauptkomponenten:")
                    # for i, var in enumerate(explained_variance):
                    #    st.write(f"Komponente {i+1}: {var:.2f}")
                elif model.method_label == 'tSNE':
                    from sklearn.manifold import trustworthiness
                    model.trustworthiness = trustworthiness(X, X_transformed)
                    # st.write(f"Trustworthiness: {trust:.2f}")
                
                # Save the model to session state
                st.session_state['trained_model'] = model
                st.session_state['predictors'] = predictors
                st.session_state['X_transformed'] = X_transformed

            if 'trained_model' in st.session_state and st.session_state['trained_model']:
                # Load the trained model from session state
                model = st.session_state['trained_model']
                predictors = st.session_state['predictors']
                # Allow user to download the serialized model
                model_file = io.BytesIO()
                joblib.dump(model, model_file)
                model_file.seek(0)

                df_modelinfo = pd.DataFrame([[get_key_from_value(supported_methods, model.method_label), model.hyperpars_str, model.id,
                                    model.date_trained.strftime("%d.%m.%Y, %H:%M"), '|'.join(predictors)]],
                                    columns=['Methode', 'Hyperparameter', 'Modell-Id', 'Trainingszeitpunkt', 'Unabhängige Variablen']
                                    )
                
                st.write('Modell-Infos:')
                st.dataframe(df_modelinfo, hide_index=True)
                
                st.download_button("Trainiertes Modell herunterladen", model_file,
                                file_name=f"{model.method_label}_{model.date_trained.strftime('%M-%H-%d-%m-%Y')}_{model.id}.pkl")

                # Create a DataFrame for the predictions
                df_Xtransformed = pd.DataFrame(st.session_state['X_transformed'], columns=[f'Dimension_{i}' for i in range(n_components)])
                
                # Display predictions
                # st.write(f"Transformierte Daten:")
                # st.dataframe(df_Xtransformed, hide_index=True)

                df_metrics = pd.DataFrame([[model.method_label, model.hyperpars_str, model.id]],
                                                columns=['Methode', 'Hyperparameter', 'Modell-Id'])
                
                if model.method_label == 'tSNE':
                    df_metrics['Trustworthiness'] = [model.trustworthiness]
                elif model.method_label == 'PCA':
                    for i, var in enumerate(model.explained_variance_ratio_):
                        df_metrics[f'Var_Komponente_{i+1}'] = [var]

                # Create a Pandas Excel writer using openpyxl as the engine
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Write predictions_df to the first sheet
                    df_Xtransformed .to_excel(writer, sheet_name='Transformierte Daten', index=False)

                    # Write the input values to second sheet
                    # new_data_df.to_excel(writer, sheet_name='Eingabewerte', index=False)

                    # Write df_modelinfo to the third sheet
                    df_modelinfo.to_excel(writer, sheet_name='Modell-Infos', index=False)

                    common_columns = df_modelinfo.columns.intersection(df_metrics.columns)
                    df_metrics.drop(columns=common_columns).to_excel(writer, sheet_name="Evaluation", index=False)
                    
                    
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
                st.dataframe(df_metrics, hide_index=True)
            