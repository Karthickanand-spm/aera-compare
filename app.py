"""Aera Compare: Streamlit entry point. Sets up pages, the shared sidebar and the footer.

Run with:  streamlit run app.py
"""

import streamlit as st

from aera.extract import ExtractionError
from ui import compare_page
from ui.sidebar import render_sidebar
from ui.state import init_state

FOOTER = "Concept prototype. Not an Aerchain product."

st.set_page_config(page_title="Aera Compare", layout="wide")
init_state()

pages = [
    st.Page(compare_page.render, title="Compare", url_path="compare", default=True),
]
page = st.navigation(pages)

try:
    render_sidebar()
    page.run()
except ExtractionError as e:  # already written in plain words
    st.error(str(e))
except Exception as e:  # never show a raw traceback
    st.error(f"Something went wrong: {e}. Try again, or reload the sample event from the sidebar.")

st.divider()
st.caption(FOOTER)
