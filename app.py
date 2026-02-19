import streamlit as st

from main.sidebar import sidebar_show
from main.views import diagnosis, efficiency, boost


def page_selected() -> None:
    PAGES = {
        "Sales Drivers": diagnosis,
        "Marketing ROI": efficiency,
        "MM Optimization": boost,
    }

    # Default page
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Sales Drivers"

    current_page = st.session_state.get("current_page", "Sales Drivers")

    if current_page in PAGES:
        PAGES[current_page].show_view()
    else:
        st.write("Select an option")


st.set_page_config(page_title="MMM Tool", layout="wide")

sidebar_show()
page_selected()
