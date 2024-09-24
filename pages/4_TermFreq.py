import streamlit as st
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = []
doc_index = []

st.title("Kodierung einer Datenmatrix aus Textdokumenten")
st.write('''Die Matrix wird als xlsx-Datei ausgegeben: 1 Spalte pro Term im Vokabular,
         1 Zeile pro Dokument. Die erste Zeile enthält die Terme im Vokabular, die erste
         Spalte enthält die ursprünglichen Dateinamen der Dokumente.''')

strip_accents = None
lowercase = False

method = st.selectbox('Wählen Sie die Kodierungsmethode:', ['One-Hot', 
                                                            'Begriffs-Häufigkeit',
                                                            'Tf-idf'
                                                            ]
                     )

st.write('''Wörter bestehend aus weniger als zwei Buchstaben werden ignoriert. Satzzeichen
         werden immer als Trennzeichen und nicht als Teil von Wörtern behandelt.\\
         Folgende Optionen betreffen die weitere Aufbereitung des Textes vor der entsprechenden
         numerischen Kodierung:''')
stripaccents = st.checkbox('Entfernen von Akzenten von Buchstaben (z.B. é->e)')
lower = st.checkbox('Normalisierung auf Kleinbuchstaben')

if stripaccents:
    strip_accents = 'unicode'

if lower:
    lowercase = True

max_features = st.number_input(label="""Beschränkung des Vokabulars auf eine Maximalzahl der häufigsten Wörter:""",
                               value=None, min_value=1,
                               placeholder="""Geben Sie eine Zahl ein... Wenn leer, werden alle identifizierten Wörter berücksichtigt""")
#  If None, will initialize empty and return None until the user provides input.

inp_files = st.file_uploader("Wählen Sie eine oder mehrere txt-Dateien",
                             accept_multiple_files=True)
for inp_file in inp_files:
    stringio = io.StringIO(inp_file.getvalue().decode("utf-8"))
    # To read file as string:
    corpus.append(stringio.read())
    doc_index.append(inp_file.name)

if len(corpus) >0:

    if method == 'One-Hot':
        vectorizer = CountVectorizer(lowercase=lowercase, strip_accents=strip_accents,
                                     binary=True, max_features=max_features)

    elif method == 'Begriffs-Häufigkeit':
        vectorizer = CountVectorizer(lowercase=lowercase, strip_accents=strip_accents,
                                     max_features=max_features)
        
    elif method == 'Tf-idf':
        vectorizer = TfidfVectorizer(lowercase=lowercase, strip_accents=strip_accents,
                                     max_features=max_features)
    else:
        vectorizer = None

    X = vectorizer.fit_transform(corpus)

    df = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out(), index=doc_index)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:  
        df.to_excel(writer, sheet_name='Term counts', index=True)

    st.download_button(
        label="xlsx-Datei herunterladen",
        data=buffer,
        file_name='term_counts.xlsx',
        mime='application/vnd.ms-excel',
    )