"""Aera Compare: Streamlit entry point. Sets up pages, the shared sidebar and the footer.

Run with:  streamlit run app.py
"""

import streamlit as st

from aera.extract import ExtractionError
from ui import ask_page, clarify_page, compare_page, create_page, decide_page
from ui.sidebar import render_sidebar
from ui.state import init_state

FOOTER = "Concept prototype. Not an Aerchain product."

st.set_page_config(page_title="Aera Compare", layout="wide")
init_state()

pages = [
    st.Page(create_page.render, title="Create RFx", url_path="create"),
    st.Page(compare_page.render, title="Compare", url_path="compare", default=True),
    st.Page(ask_page.render, title="Ask", url_path="ask"),
    st.Page(decide_page.render, title="Decide", url_path="decide"),
    clarify_page.page(),
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
