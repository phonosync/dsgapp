import io
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

import utils

st.title("Berechnung der Paarweisen Distanzmatrix")
st.write("In der Berechnung der Distanzen zwischen zwei Vektoren werden jeweils nur die gemeinsam besetzten Elemente berücksichtigt.")

def pairwise_distances_nan(X, metric='euclidean'):
    """
    Calculate pairwise distances between rows of X, ignoring NaN values.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        Input data.
    metric : str or callable, default='euclidean'
        The distance metric to use.
    
    Returns:
    D : array, shape (n_samples, n_samples)
        Pairwise distance matrix.
    """
    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i, n_samples):
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])
            if np.any(mask):
                D[i, j] = D[j, i] = pairwise_distances([X[i][mask]], [X[j][mask]], metric=metric)[0]
            else:
                D[i, j] = D[j, i] = np.nan  # If no valid elements, set distance to NaN
    
    return D


# select metric
metric = st.selectbox('Wähle die zu berechnende Distanz-Metrik',
                            ['cityblock', 'cosine', 'euclidean', 'l1',
                                'l2', 'manhattan']
                        )

if metric == 'cosine':
    st.warning('''`cosine` berechnet nicht die Kosinus-Ähnlichkeit, sondern
                die Kosinus-Distanz zwischen zwei Vektoren 
            $=1-\\frac{\mathbf{u}\cdot \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}$''',icon="⚠️")

st.write('''Lade eine csv-Datei mit den Daten hoch. Ein Sample pro Zeile.\\
            Die erste Reihe kann als Kopfzeile mit Merkmal-Bezeichnungen interpretiert werden.\\
            Die erste Spalte kann als Sample-Indices interpretiert werden. Diese werden, wenn vorhanden, als Spalten- und Zeilen-Labels im Output-File verwendet.
         ''')
# df_sample_inp = pd.DataFrame({'Variable 1': [0.0, 4.1, 2.3], 'Variable 2': [0, 1, 1], 
#                                'Variable 3': [5, 3, 9]})
# st.markdown(df_sample_inp.style.hide(axis="index").to_html(), unsafe_allow_html=True)

st.write("""Sample-Indices können in der ersten Spalte des hochgeladenen Datei 
            angegeben werden. Sie werden als Spalten- und Zeilen-Labels im Output-File 
            verwendet. Wähle in der folgenden Checkbox:""")

df_sample_inp = pd.DataFrame({'Index': ['Sample 1', 'Sample 2', 'Sample 3'], 
                                'Merkmal 1': [0.0, 4.1, 2.3], 'Merkmal 2': [0, 1, 1], 
                                'Merkmal 3': [5, 3, 9]})

st.dataframe(df_sample_inp, hide_index=True)

# st.markdown(df_sample_inp.style.hide(axis="index").to_html(), unsafe_allow_html=True)

index_column = st.checkbox('Erste Spalte enthält Sample-Indizes')
header_row = st.checkbox('Erste Zeile enthält Spalten-Labels') 

inp_file = st.file_uploader("Wähle eine csv-Datei")

if inp_file is not None:

    df = utils.inp_file_2_csv(inp_file=inp_file, index_column=index_column,
                                header_row=header_row)
    

    # if index_column:
    #     df.set_index(df.columns[0],inplace=True)

    X = df.to_numpy(copy=True)

    # D = pairwise_distances(X, metric=metric)
    D = pairwise_distances_nan(X, metric=metric)

    df_dists = pd.DataFrame(data=D,
                            index=df.index,
                            columns=df.index
                            )

    df_pars = pd.DataFrame([['metric', 'date'],[metric, 'today']])

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:  
        df_dists.to_excel(writer, sheet_name='Distanzmatrix', index=True)
        df_pars.to_excel(writer, sheet_name='Berechnungsmethode', index=False)


    st.download_button(
        label="Distanzmatrix als xlsx-File herunterladen",
        data=buffer,
        file_name='distance_matrix.xlsx',
        mime='application/vnd.ms-excel',
    )