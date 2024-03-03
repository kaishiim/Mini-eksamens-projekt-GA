import streamlit as st
from streamlit_timeline import st_timeline

st.set_page_config(layout="wide")

items = [
    {"id": 1, "content": "Picture distortion algorithm", "start": "2023-03-30"},
    {"id": 2, "content": "SOP LV-model visualzation", "start": "2023-12-18"},
    {"id": 3, "content": "SOP optimization algorithm", "start": "2023-12-19"},
    {"id": 4, "content": "TSP-Gentic algorithm", "start": "2024-02-12"},
]

timeline = st_timeline(items, groups=[], options={}, height="300px")
st.subheader("Selected item")
st.write(timeline)