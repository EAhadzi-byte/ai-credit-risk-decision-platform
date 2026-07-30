"""
=========================================================
Professional KPI Cards
=========================================================
"""

import streamlit as st


def display_metrics(prediction):

    probability = prediction["default_probability"]
    confidence = prediction["confidence"]
    risk = prediction["risk_class"]

    if risk == "Low Risk":
        colour = "#27AE60"
        icon = "🟢"
    else:
        colour = "#E74C3C"
        icon = "🔴"

    st.markdown(
        f"""
<style>

.metric-card {{

background:white;
padding:18px;
border-radius:15px;
box-shadow:0 4px 10px rgba(0,0,0,.08);
border-left:8px solid {colour};
margin-bottom:15px;}}

.metric-title {{

font-size:16px;
color:#666;}}

.metric-value {{

font-size:34px;
font-weight:bold;}}

</style>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-title">

Probability of Default

</div>

<div class="metric-value">

{probability:.2%}

</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-title">

Model Confidence

</div>

<div class="metric-value">

{confidence:.2%}

</div>

</div>
""",
            unsafe_allow_html=True,)

    with c3:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-title">

Risk Classification

</div>

<div class="metric-value">

{icon} {risk}

</div>

</div>
""",unsafe_allow_html=True,)