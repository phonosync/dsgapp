import streamlit as st

# Sidebar navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis"])

# Load the selected page
# if page == "Home":
#     st.title("Welcome to the Streamlit Multipage App")
#    st.write("Use the sidebar to navigate to different pages.")
#elif page == "Exploratory Data Analysis":
#    from pages import eda
#    eda.show()
st.set_page_config(
    page_title="DSG-Apps",
    page_icon="👋",
)

st.title("DSG-Apps")
st.write("Eine Sammlung von Apps zur Illustration einiger zentralen Operationen in einem Data Science-Projekt. Über die Seitenleiste sind die verschiedenen Funktionalitäten zugänglich.")


