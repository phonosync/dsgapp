import streamlit as st
import io
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

model = None

st.title("*k* Nächste Nachbarn-Klassifikation")

st.write('''Basierend auf Trainingsdaten, welche als csv-Datei hochgeladen
         werden, wird ein *k*-Nächster Nachbar-Algorithmus "trainiert". Dieser kann
         für Vorhersagen auf neuen Daten (welche in einem zweiten csv-File hochgeladen werden)
          genutzt werden.''')

if not model:
    st.write("""Laden Sie die Trainingsdaten als csv-Datei in folgendem Format hoch:\\
             Erste Zeile: Spaltenbezeichnungen. 
             Eine Spalte pro Feature (unabhängige Variablen) und für die
              Zielvariable (Labels, die es mit dem zu trainierenden Modell vorherzusagen gilt).
             Alle Einträge müssen in numerischer Form kodiert vorliegen. Beispiel mit 3 Features, 
             der Zielvariable (insgesamt 4 Spalten) und 4 Samples (mit der Kopzeile insgesamt 5 Zeilen):""")
    df = pd.DataFrame({'Feature 1': [0.0, 4.1, 2.3, 1.9], 'Feature 2': [0, 1, 1, 0], 
                              'Feature 3': [5, 3, 9, 2], 'Target': [0, 1, 1, 1]}
                              )
    
    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(data=df)

    train_file = st.file_uploader("Wählen Sie eine Datei")
    if train_file is not None:
        
        # We always assume column labels (see format description above)
        header = 0

        bla = io.StringIO(train_file.getvalue().decode("utf-8"))
        dialect = csv.Sniffer().sniff(bla.read())
        df_train = pd.read_csv(train_file, dialect=dialect, header=header)

        # select dependent variable
        target_col = st.selectbox('Wählen Sie die Zielvariable (Spalte mit Labels)',
                              df_train.columns
                             )

        independent_cols = df_train.drop(target_col, axis=1).columns

        n_neighbors = st.number_input("Wählen Sie *k*: Anzahl zu berücksichtigender Nachbarn", 
                                      min_value=0, format='%i', value=3
                                     )
        
        metric = st.selectbox('Wählen Sie die zu berechnende Distanz-Metrik',
                                ['cosine', 'euclidean',
                                 'manhattan']
                         )
        
        if metric == 'cosine':
            st.warning('''`cosine` berechnet nicht die Kosinus-Ähnlichkeit, sondern
                       die Kosinus-Distanz zwischen zwei Vektoren
                       $=1-\\frac{\mathbf{u}\cdot \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}$''',
                       icon="⚠️"
                    )

        # train model
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        model.fit(df_train.drop(target_col, axis=1), df_train[target_col])
        
if not model:
    status = ":red[MODELL NICHT TRAINIERT]"
else:
    status = ":green[MODELL TRAINIERT]"

st.markdown(status)

if model:
    st.write("""Laden Sie eine csv-Datei mit Samples hoch, für die Sie mithilfe des trainierten Modells 
             Vorhersagen machen möchten. Erste Zeile: Spaltenbezeichnungen. Alle im Training verwendeten 
             Features müssen als separate Spalte vorhanden sein. Übrige Spalten werden ignoriert.""")
    test_file = st.file_uploader("Wählen Sie eine Datei", key='test_file_uploader')
    if test_file is not None:
        # We always assume column labels (see format description above)
        header = 0

        bla = io.StringIO(test_file.getvalue().decode("utf-8"))
        dialect = csv.Sniffer().sniff(bla.read())
        df_test = pd.read_csv(test_file, dialect=dialect, usecols=independent_cols, header=header)

        y_hat = model.predict(df_test.values)

        df_y_hat = pd.DataFrame({target_col: y_hat}) # pd.concat([df_test, pd.DataFrame({target_col: y_hat})], axis=1)

        # csv = bla.to_csv(sep=';', index=False, encoding='utf-8')

        # xlsx = bla.to_excel('knn_class_vorhersagen.xlsx', sheet_name='Vorhersagen')

        (par_keys, par_values) = zip(*model.get_params().items())
        df_pars = pd.DataFrame({'Model Parameters': par_keys, 'Value': par_values})
        

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:  
            df_y_hat.to_excel(writer, sheet_name='Vorhersagen', index=False)
            df_test.to_excel(writer, sheet_name='Samples für die Vorhersagen', index=False)
            df_train.to_excel(writer, sheet_name='Training Samples', index=False)
            df_pars.to_excel(writer, sheet_name='Modellparameter', index=False)
        
        st.download_button(
            label="Vorhersagen als xlsx-Datei herunterladen",
            data=buffer,
            file_name='{}nn_vorhersagen.xlsx'.format(n_neighbors),
            mime='application/vnd.ms-excel',
        )