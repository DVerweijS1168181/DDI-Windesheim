from io import BytesIO
import PyPDF2
import pandas as pd
import streamlit as st


def count_pages(pdf_bytes):
    pdf_file = BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    return num_pages


def remove_pages(pdf_bytes, pages_to_remove):
    pdf_writer = PyPDF2.PdfWriter()
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

    for page_num, page in enumerate(pdf_reader.pages):
        if page_num + 1 not in pages_to_remove:
            pdf_writer.add_page(page)

    output_pdf = BytesIO()
    pdf_writer.write(output_pdf)
    return output_pdf.getvalue()


def filtering_pages():
    if not st.session_state.get("file_details", []):
        st.sidebar.warning(
            "Upload eerst documenten voordat u pagina's kunt filteren.")
        return None

    file_details = st.session_state.get("file_details", [])
    pages_per_document = {}

    for file_info in file_details:
        file_name = file_info["File_Name"]
        num_pages = file_info["Amount_Pages"]
        pages_per_document[file_name] = [
            f"Pagina {j}" for j in range(1, num_pages + 1)]

    selected_documents = st.sidebar.multiselect(
        "Selecteer documenten", [file_info["File_Name"] for file_info in file_details], key="selected_documents",
        placeholder="Selecteer documenten")

    selected_pages = {}
    if selected_documents:
        for document in selected_documents:
            pagina_keuze = st.sidebar.multiselect(
                f"Selecteer pagina's voor {document}", pages_per_document[document], key=f"pages_to_remove_{document}",
                placeholder=f"Selecteer pagina's")
            selected_pages[document] = [
                int(page.split()[1]) for page in pagina_keuze]

    st.session_state["selected_pages"] = selected_pages


def reset_session():
    if st.button("Reset", type="primary"):
        st.session_state.clear()
        st.experimental_rerun()
