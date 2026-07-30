"""
=========================================================
Professional Dashboard Theme
=========================================================
"""

import streamlit as st


def apply_styles():

    st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#F5F7FA;
}

/* Headings */
h1{
    color:#003366 !important;
    font-weight:700;
}

h2{
    color:#003366 !important;
}

h3{
    color:#003366 !important;
}

/* Force normal text to be dark */
p, li, span, label{
    color:#222222 !important;
}

/* Markdown text */
div[data-testid="stMarkdownContainer"]{
    color:#222222 !important;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    border:1px solid #D9D9D9;
    padding:18px;
    box-shadow:0 3px 8px rgba(0,0,0,.08);
}

/* Buttons */
div.stButton > button{
    border-radius:10px;
    background:#003366;
    color:white;
    font-weight:bold;
}

div.stButton > button:hover{
    background:#C8102E;
}

/* Download Button */
div.stDownloadButton > button{
    border-radius:10px;
}

/* Alerts */
div[data-testid="stAlert"]{
    border-radius:12px;
}

</style>
""",
    unsafe_allow_html=True)