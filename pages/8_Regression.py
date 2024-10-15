import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib
import io
import utils

st.title('Regressionsmethoden')

@st.cache_data
def load_data(uploaded_file):
    return utils.inp_file_2_csv(inp_file=uploaded_file, index_column=False,
                                  header_row=True)

# File uploader for training data
train_file = st.file_uploader("Wähle eine CSV-Datei mit den Trainingsdaten", type="csv")


if train_file is not None:
    # Read the CSV file
    df = load_data(train_file)
    
    # Display the DataFrame
    st.write("Hochgeladene Trainingsdaten:")
    st.write(df[:5])
    
    # Consistency check: no missing values, only numeric values allowed
    if df.isnull().values.any():
        st.error("Die Daten enthalten fehlende Werte. Bitte bereinigen Sie die Daten und laden Sie sie erneut hoch.")
    elif not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        st.error("Die Daten enthalten nicht-numerische Werte. Bitte bereinigen Sie die Daten und laden Sie sie erneut hoch.")
    else:
        # Extract columns
        columns = df.columns.tolist()
        
        # User selects predictors and target variable
        predictors = st.multiselect("Wähle die unabhängigen Merkmale, die im Training berücksichtig werden sollen", columns)
        target = st.selectbox("Wähle die abhängige Variable (Zielvariable)", columns)
        
        if predictors and target:
            # User selects regression algorithm
            algorithm = st.selectbox("Wähle den Regressionsalgorithmus", ["Lineare Regression", "Polynomiale Regression", "Nächste Nachbarn-Heuristik"])
            
            # Train the regression model
            X = df[predictors]
            y = df[target]
            
            if algorithm == "Lineare Regression":
                model = LinearRegression()
            elif algorithm == "Polynomiale Regression":
                degree = st.number_input("Wähle den Grad des Polynoms", min_value=1, value=2)
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            elif algorithm == "Nächste Nachbarn-Heuristik":
                n_neighbors = st.number_input("Wähle die Anzahl der Nachbarn (k)", min_value=1, value=3)
                metric = st.selectbox("Wähle die Ähnlichkeitsmetrik", ["manhattan", "euclidean", "cosine"])
                model = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)
            
            
            model.fit(X, y)
            st.success("Das Modell wurde erfolgreich trainiert.")
            
            # Allow user to download the serialized model
            model_file = io.BytesIO()
            joblib.dump(model, model_file)
            model_file.seek(0)
            st.download_button("Modell herunterladen", model_file, file_name="trained_model.pkl")
            
            # File uploader for test data
            test_file = st.file_uploader("Wähle eine CSV-Datei mit den Testdaten", type="csv")

            if test_file is not None:
                # Read the CSV file
                test_df = load_data(train_file)
                
                # Display the DataFrame
                st.write("Hochgeladene Testdaten:")
                st.write(test_df[:5])
                
                # Consistency check: no missing values, only numeric values allowed, and all columns used in training are present
                if test_df.isnull().values.any():
                    st.error("Die Testdaten enthalten fehlende Werte. Bitte bereinigen Sie die Daten und laden Sie sie erneut hoch.")
                elif not all(test_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                    st.error("Die Testdaten enthalten nicht-numerische Werte. Bitte bereinigen Sie die Daten und laden Sie sie erneut hoch.")
                elif not all(col in test_df.columns for col in predictors):
                    st.error("Die Testdaten enthalten nicht alle für das Training verwendeten Spalten. Bitte überprüfen Sie die Daten und laden Sie sie erneut hoch.")
                else:
                    # Make predictions
                    X_test = test_df[predictors]
                    y_test = test_df[target]
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    msd = mean_squared_error(y_test, y_pred)
                    rmsd = np.sqrt(msd)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Display metrics
                    # Create DataFrame 1
                    df_metrics = pd.DataFrame({
                        "Metrik": ["MSE", "RMSD", "MAE"],
                        "Wert": [msd, rmsd, mae]
                    })
                    st.write("Evaluationsmetriken berechnet für die Vorhersagen auf den Testdaten:")
                    st.dataframe(df_metrics, hide_index=True)