import streamlit as st
import pytesseract
import PIL
import pandas as pd
import io
import zipfile

st.title("Texterkennung")

uploaded_files = st.file_uploader("Wähle eine oder mehrere Bilddatei(en)", accept_multiple_files=True)

# Language selection
languages = ["None", "deu", "eng", "fra", "ita", "spa", "por"]  # Add more languages as needed
selected_language = st.selectbox("Wähle eine Sprache für die Texterkennung. Falls unbekannt: None", languages)


if uploaded_files and selected_language:
    if st.button("Texterkennung starten"):
        extracted_texts = []
        for uploaded_file in uploaded_files:
            # Open the uploaded image
            try:
                img = PIL.Image.open(uploaded_file)
                # st.image(img, caption='Uploaded Image.', use_column_width=True)
            except PIL.UnidentifiedImageError:
                st.error("Unbekanntes Bildformat. Bitte lade eine gültige Bilddatei hoch.")
                continue  # Skip to the next file if an error occurs
            
            # Perform OCR using Tesseract
            if selected_language != "None":
                text = pytesseract.image_to_string(img, lang=selected_language)
            else:
                text = pytesseract.image_to_string(img)  # Default language setting
            
            
            extracted_texts.append((uploaded_file.name, text))
            # st.write("Extracted Text:")
            # st.write(text)
        
        if len(uploaded_files) == 1:
            # Save as a .txt file
            file_name = uploaded_files[0].name.rsplit('.', 1)[0] + '.txt'
            with io.StringIO() as txt_buffer:
                txt_buffer.write(extracted_texts[0][1])
                txt_data = txt_buffer.getvalue()
            st.download_button(label="Text-Datei herunterladen", data=txt_data, file_name=file_name, mime="text/plain")
        else:

            # Save each file as a .txt file and add to ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for file_name, text in extracted_texts:
                    txt_file_name = file_name.rsplit('.', 1)[0] + '.txt'
                    with io.StringIO() as txt_buffer:
                        txt_buffer.write(text)
                        txt_data = txt_buffer.getvalue()
                    zip_file.writestr(txt_file_name, txt_data)

            # Provide download button for the ZIP file
            st.download_button(
                label="Zip-Archiv mit allen Text-Dateien herunterladen",
                data=zip_buffer.getvalue(),
                file_name="extracted_texts.zip",
                mime="application/zip"
            )