import streamlit as st
import io
import numpy as np
import pandas as pd
import Levenshtein
import utils

st.title("Levenshtein-Distanz")

st.write('''Die Editierdistanz wird zwischen den Wordpaaren in den ersten beiden Spalten
         der Input csv-Datei berechnet. Die Resultate werden in Form einer paarweisen Distanz-
         Matrix in einem xlsx-File ausgegeben.''')

# strip_accents = None
# lowercase = False

# stripaccents = st.checkbox('Strip accents')
# lower = st.checkbox('Convert to lowercase')

#if stripaccents:
#     strip_accents = 'unicode'

# if lower:
#    lowercase = True

header_row = st.checkbox('Input-Datei enthält Kopfeile (wird in der Distanzberechnung nicht berücksichtigt)')

st.write('''Den 3 erlaubten Editier-Operationen können unterschiedliche Gewichtungen in der
         Distanz-Berechnung zugewiesen werden:''')
col1, col2, col3 = st.columns(3)
with col1:
    weight_ins = st.number_input(label='Gewicht für Einfügen', value=1)

with col2:
    weight_del = st.number_input(label='Gewicht für Löschen', value=1)

with col3:
    weight_sub = st.number_input(label='Gewicht für Ersetzen', value=1)

# only_pairwise = st.checkbox('''Calculate levenshtein distance only between pairs of words in 
#                             the same row instead of all unique pairs of words in the first 
#                             two columns.''')

inp_file = st.file_uploader("Wählen Sie eine csv-Datei")

if inp_file is not None:
        
    if header_row:
        header = 0
    else:
        header = None

    df_inp = utils.inp_file_2_csv(inp_file=inp_file, index_column=False, header_row=header_row)
    
    bla = list(df_inp[~df_inp[df_inp.columns[0]].isna()][df_inp.columns[0]].values)

    blue = list(df_inp[~df_inp[df_inp.columns[1]].isna()][df_inp.columns[1]].values)

    a = np.zeros(shape=(len(bla),len(blue)), dtype=int)

    df_dists = pd.DataFrame(data=a, index=bla, columns=blue)

    for row in df_dists.itertuples():
        for s2 in df_dists.columns:
            df_dists.loc[row.Index][s2] = Levenshtein.distance(row.Index, s2, 
                                                               weights=(weight_ins,
                                                                        weight_del,
                                                                        weight_sub
                                                                        ), 
                                                               processor=None, 
                                                               score_cutoff=None
                                                               )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:  
        df_dists.to_excel(writer, sheet_name='Levenshtein distances', index=True)

    st.download_button(
        label="Levenshtein-Distanzen als xlsx-Datei herunterladen",
        data=buffer,
        file_name='levenshtein_dists.xlsx',
        mime='application/vnd.ms-excel',
    )