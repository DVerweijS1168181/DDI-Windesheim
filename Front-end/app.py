import PyPDF2
import pandas as pd
import streamlit as st
from functions_widgets import reset_session, filtering_pages, count_pages
from functions_network import *
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(page_title="Streamlit App",
                   page_icon=":skull_and_crossbones:", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Data Gedreven Gebiedsontwikkeling")

# download_models() # Download the models for the first time (NL, EN, stopwords, etc)

# Initialize session state variables


def initialize_session_state():
    variables_to_initialize = {
        'button_priority': False,
        'button_filter_pagina': False,
        'button_choose_theme': False,
        'button_top_words': False,
        'button_redo_priority': False,
        'button_generate_network': False,
        'base_priority': "Middel",
        'selected_pages': {},
        'theme_selection': "Geen specificatie",
        'language': "NL",
        'top_words': 300,
        'details_network': 0,
        'network_html': None

    }
    for var_name, default_value in variables_to_initialize.items():
        if var_name not in st.session_state:
            st.session_state[var_name] = default_value


initialize_session_state()


def file_uploading():
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    if "file_details" not in st.session_state:
        st.session_state["file_details"] = []

    files = st.sidebar.file_uploader(
        "Upload PDF-documenten", type="pdf", accept_multiple_files=True,
        key=st.session_state["file_uploader_key"],
        help="Let op: Alleen PDF-documenten kunnen worden geüpload, wanneer documenten met dezelfde naam worden gegeven worden dezelfde prioriteiten gegeven aan de documenten."
    )

    if files:
        for file in files:
            existing_file_index = next((i for i, f in enumerate(
                st.session_state["uploaded_files"]) if f.name == file.name), None)
            if existing_file_index is not None:
                del st.session_state["uploaded_files"][existing_file_index]
                st.session_state["file_details"] = [
                    f for f in st.session_state["file_details"] if f["File_Name"] != file.name]

            st.session_state["uploaded_files"].append(file)
            num_pages = count_pages(file.getvalue())
            file_details = {
                "File_Name": file.name,
                "Weight": "Middel",
                "Amount_Pages": num_pages,
            }
            st.session_state["file_details"].append(file_details)


def choose_theme():
    if not st.session_state.get("file_details", []):
        st.sidebar.empty()
        return None
    thema_options = ["Geen specificatie",
                     "Water", "Mobiliteit", "Duurzaamheid"]
    theme_selection = st.sidebar.selectbox(
        "Kies thema", thema_options, label_visibility="collapsed")
    st.session_state.theme_selection = theme_selection


def choose_language():
    if not st.session_state.get("file_details", []):
        st.sidebar.empty()
        return None
    language_options = ["NL", "EN"]
    language_selection = st.sidebar.selectbox(
        "Kies taal", language_options, label_visibility="collapsed")
    st.session_state.language = language_selection


def choose_top_words():
    if not st.session_state.get("file_details", []):
        st.sidebar.empty()
        return None
    words = st.sidebar.number_input(
        "Woorden", min_value=50, max_value=500, value=300, step=50, label_visibility="collapsed")
    st.session_state.top_words = words


def priority_form():
    if not st.session_state.get("uploaded_files"):
        st.sidebar.empty()
        return

    valid_options = ["Laag", "Middel", "Hoog"]
    base_priority = st.session_state.get("base_priority", "Middel")
    if base_priority not in valid_options:
        base_priority = "Middel"

    priorities = {}
    for i, file in enumerate(st.session_state["uploaded_files"]):
        name_short = file.name[:20] + \
            "..." if len(file.name) > 20 else file.name
        st.sidebar.write(f"Prioriteit voor {name_short}")
        priority_key = f"priority_slider_{i}"
        priority = st.sidebar.select_slider(
            f"{file.name}", options=valid_options, key=priority_key, label_visibility="collapsed", value=base_priority)
        priorities[file.name] = priority

    for file_name, priority in priorities.items():
        for file_detail in st.session_state["file_details"]:
            if file_detail["File_Name"] == file_name:
                file_detail["Weight"] = priority


def display_network():
    try:
        with open('water_network.html', 'r') as f:
            html = f.read()
            st.session_state.network_html = html
            st.components.v1.html(html, scrolling=True, height=1200)
    except Exception as e:
        st.error(f"Error loading network HTML: {e}")


def get_uploaded_file_names():
    return [file["File_Name"] for file in st.session_state["file_details"]]


def get_uploaded_file_weights():
    if "file_details" in st.session_state:
        return [details["Weight"] for details in st.session_state["file_details"]]
    return []


def get_selected_pages():
    return st.session_state.get("selected_pages", {})


def get_theme_selection():
    return st.session_state.get("theme_selection", "Geen specificatie")


def get_top_words():
    return st.session_state.get("top_words", 300)


def get_language():
    return st.session_state.get("language", "NL")


def display_community_info():
    if 'community_df' not in st.session_state:
        st.warning("Geen community data beschikbaar.")
        return

    community_df = st.session_state.community_df
    st.header("Thema's")
    grouped = community_df.groupby('Community')
    grouped_list = list(grouped)

    thema_counter = 1  # Initialize the counter to start from 1

    for i in range(0, len(grouped_list), 2):
        community1, group1 = grouped_list[i]
        leader1 = group1['Leader'].iloc[0] if not group1.empty else "Unknown"
        color1 = group1['Color'].values[0] if not group1.empty else "N/A"

        # Use uppercased leader name for subheader
        st.subheader(f"{leader1.capitalize()}")
        st.write(f"Thema {thema_counter} | Kleur: {color1}")
        with st.expander(f"Gegevens van thema {thema_counter}", expanded=False):
            st.dataframe(group1[['Node']], height=400, width=800)

        thema_counter += 1  # Increment the counter after processing the first theme

        if i + 1 < len(grouped_list):
            community2, group2 = grouped_list[i + 1]
            leader2 = group2['Leader'].iloc[0] if not group2.empty else "Unknown"
            color2 = group2['Color'].values[0] if not group2.empty else "N/A"

            st.subheader(f"{leader2.capitalize()}")
            st.write(f"Thema {thema_counter} | Kleur: {color2}")
            with st.expander(f"Gegevens van thema {thema_counter}", expanded=False):
                st.dataframe(group2[['Node']], height=400, width=800)

            thema_counter += 1  # Increment the counter after processing the second theme


def generate_network():
    if not st.session_state.get("file_details", []):
        st.warning("Upload eerst documenten voordat u een netwerk kan genereren.")
        return None

    pdf_files = get_uploaded_file_names()
    doc_weights = {file["File_Name"]: 0.5 if file["Weight"] == "Laag" else 1.0 if file["Weight"] == "Middel" else 1.5
                   for file in st.session_state["file_details"]}
    thema = get_theme_selection()
    filter_pages = get_selected_pages()

    if thema == "Geen specificatie":
        relations_df, _ = main(pdf_files, doc_weights,
                               get_top_words(), filter_pages, get_language())
        relations_df.columns = ['Word Pair', 'Source', 'Target', 'Count',
                                'Source_Occurrence', 'Target_Occurrence', 'Edge_Weight']
    else:
        relations_df, _ = main_cat(
            pdf_files, doc_weights, thema, get_top_words(), filter_pages, get_language())
        relations_df.columns = ['Word Pair', 'Source', 'Source_Category', 'Target',
                                'Target_Category', 'Count', 'Source_Occurrence', 'Target_Occurrence', 'Edge_Weight']

    community_df, html_file = create_network(relations_df)
    with open(html_file, 'r') as f:
        html = f.read()
        st.session_state.network_html = html

    st.session_state.community_df = community_df

    # display_community_info()


col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.markdown('<p style="font-size: 60px; font-weight: bold;">InsightNet</p>',
                unsafe_allow_html=True)
    display_network()
with col2:
    # with st.expander("See session states", expanded=True):
    #     st.write("st.session_state object:", st.session_state)
    if 'community_df' in st.session_state:
        display_community_info()

st.sidebar.title("Upload documenten")
file_uploading()

colSidebar1, colSidebar2 = st.sidebar.columns(2, gap="Large")
with colSidebar1:
    reset_session()
with colSidebar2:
    st.empty()

st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <h1 style="text-decoration: underline; margin-bottom: 5px;">Parameters</h1>
    </div>
    <hr style="margin-top: 5px;">
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(" ## Taal")
choose_language()
st.sidebar.markdown("<hr style='margin-top: 5px;'>", unsafe_allow_html=True)
st.sidebar.markdown(" ## Prioriteer documenten")
priority_form()

st.sidebar.markdown("<hr style='margin-top: 5px;'>", unsafe_allow_html=True)
st.sidebar.markdown(" ## Filter gewenste pagina's")
if st.session_state.uploaded_files:
    filtering_pages()

st.sidebar.markdown("<hr style='margin-top: 5px;'>", unsafe_allow_html=True)
st.sidebar.markdown(" ## Kies thema")
choose_theme()

st.sidebar.markdown("<hr style='margin-top: 5px;'>", unsafe_allow_html=True)
st.sidebar.markdown(" ## Kies aantal top woorden")
choose_top_words()
st.sidebar.markdown("<hr style='margin-top: 5px;'>", unsafe_allow_html=True)

col1side, col2side, col3side = st.sidebar.columns([0.25, 0.5, 0.25])
with col1side:
    st.empty()
with col2side:
    st.button("Genereer netwerk", on_click=generate_network, type="primary")
    if st.session_state.button_generate_network:
        generate_network()


footer = """<style>
    a:link,
    a:visited {
        color: blue;
        background-color: transparent;
        text-decoration: underline;
    }
    a:hover,
    a:active {
        color: red;
        background-color: transparent;
        text-decoration: underline;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        font: bolder 18 arial;
    }
</style>
<div class="footer">
    <p>Developed with ❤ by <a style='display: block; text-align: center;'
            href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank">Team: Data Gedreven Gebiedsontwikkeling</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
