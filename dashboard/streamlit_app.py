"""
=========================================================
Explainable AI Credit Risk Platform
=========================================================
"""

# Imports
# _________________________________________________________

import streamlit as st

from api_client import (predict_loan,explain_loan)

from gauge import risk_gauge

from cards import display_metrics

from charts import shap_bar_chart

from summary import display_summary

from styles import apply_styles

from report import create_pdf

# Page Configuration
# _______________________________________________________

st.set_page_config(

    page_title="Explainable AI Credit Risk Platform",

    page_icon="💳",

    layout="wide",

    initial_sidebar_state="expanded")


# Apply Dashboard Theme
# ________________________________________________________

apply_styles()


# Session State
# ________________________________________________________

if "prediction" not in st.session_state:

    st.session_state["prediction"] = None


if "explanation" not in st.session_state:

    st.session_state["explanation"] = None



# Header
# __________________________________________________________

st.title("💳 Explainable AI Credit Risk Decision Engine")

st.markdown(
"""
This platform predicts the probability of loan default using an
optimised **XGBoost** model enhanced with **macroeconomic indicators**
and provides **Explainable AI (SHAP)** explanations for every prediction.
""")

st.divider()

st.markdown("""
<div style="
background:#FFFFFF;
padding:30px;
border-radius:20px;
border-left:8px solid #014421;
box-shadow:0 8px 20px rgba(0,0,0,0.08);
margin-bottom:25px;
">

<h2 style="
color:#014421;
margin-top:0;
margin-bottom:10px;
font-size:36px;
font-weight:700;
">

🤖 AI-Powered Explainable Credit Decision Engine

</h2>

<p style="
color:#444444;
font-size:18px;
line-height:1.8;
margin-bottom:20px;
">

An enterprise-grade platform that combines machine learning,
macroeconomic intelligence and Explainable AI to deliver
transparent, trustworthy and auditable credit risk decisions.

</p>

<div style="
display:flex;
flex-wrap:wrap;
gap:10px;
margin-bottom:20px;
">

<span style="
background:#E8F5E9;
color:#014421;
padding:8px 16px;
border-radius:20px;
font-weight:600;
">
✓ XGBoost
</span>

<span style="
background:#E3F2FD;
color:#1565C0;
padding:8px 16px;
border-radius:20px;
font-weight:600;
">
✓ SHAP Explainability
</span>

<span style="
background:#FFF8E1;
color:#F57C00;
padding:8px 16px;
border-radius:20px;
font-weight:600;
">
✓ FRED Macroeconomic Data
</span>

<span style="
background:#F3E5F5;
color:#6A1B9A;
padding:8px 16px;
border-radius:20px;
font-weight:600;
">
✓ FastAPI
</span>

<span style="
background:#E0F7FA;
color:#006064;
padding:8px 16px;
border-radius:20px;
font-weight:600;
">
✓ Streamlit
</span>

</div>

<p style="
color:#666666;
font-size:16px;
margin-bottom:0;
">

Designed for financial institutions, fintech companies and
automotive finance providers to support explainable, data-driven
credit decisions.

</p>

</div>
""", unsafe_allow_html=True)

# Sidebar
# __________________________________________________________

st.sidebar.header("Loan Application")


loan_amnt = st.sidebar.number_input(

    "Loan Amount (£)",

    min_value=1000,

    max_value=50000,

    value=10000,

    step=500)


annual_inc = st.sidebar.number_input(

    "Annual Income (£)",

    min_value=1000,

    max_value=300000,

    value=50000,

    step=1000)


int_rate = st.sidebar.slider(

    "Interest Rate (%)",

    5.0,

    35.0,

    12.0)


dti = st.sidebar.slider(

    "Debt-to-Income Ratio",

    0.0,

    50.0,

    15.0)


revol_util = st.sidebar.slider(

    "Revolving Credit Utilisation (%)",

    0.0,

    150.0,

    45.0)


term = st.sidebar.selectbox(

    "Loan Term",

    [

        "36 months",

        "60 months"

    ])


grade = st.sidebar.selectbox(

    "Loan Grade",

    [

        "A",

        "B",

        "C",

        "D",

        "E",

        "F",

        "G"])


home_ownership = st.sidebar.selectbox(

    "Home Ownership",

    [

        "MORTGAGE",

        "OWN",

        "RENT"])


title = st.sidebar.text_input(

    "Loan Purpose",

    "Debt Consolidation")


# kept fixed because  macro features
# are mapped to this period

issue_d = "2018-03"


# Build Request
# __________________________________________________

application = {

    "loan_amnt": loan_amnt,

    "term": term,

    "grade": grade,

    "int_rate": int_rate,

    "annual_inc": annual_inc,

    "dti": dti,

    "home_ownership": home_ownership,

    "revol_util": revol_util,

    "title": title,

    "issue_d": issue_d}


# Predict Button
# __________________________________________________________

if st.sidebar.button("🚀 Predict Credit Risk"):

    with st.spinner("Running Explainable AI model..."):

        try:

            prediction = predict_loan(application)

            explanation = explain_loan(application)

            st.session_state["prediction"] = prediction

            st.session_state["explanation"] = explanation

        except Exception as e:

            st.error(f"Prediction failed: {e}")



# Retrieve Results
# ______________________________________________________________

prediction = st.session_state["prediction"]

explanation = st.session_state["explanation"]


# Display Prediction Results
# ______________________________________________________________

if prediction is not None:

    st.divider()

    st.header("Prediction Results")

    
    # Risk Gauge
    # ------------------------------------------

    gauge = risk_gauge(
        prediction["default_probability"],prediction["risk_class"])

    st.plotly_chart(gauge,use_container_width=True)

    
    # KPI Cards
    # ------------------------------------------

    display_metrics(prediction)

    st.write("")

    
    # Risk Status Banner
    # ------------------------------------------

    if prediction["risk_class"] == "Low Risk":

        st.success(
            f"""
### ✅ Low Risk Borrower

Estimated Probability of Default:
**{prediction['default_probability']:.2%}**

Model Confidence:
**{prediction['confidence']:.2%}**
""")

    else:

        st.error(
            f"""
### ⚠ High Risk Borrower

Estimated Probability of Default:
**{prediction['default_probability']:.2%}**

Model Confidence:
**{prediction['confidence']:.2%}**
""")

    # Borrower Summary
    # ------------------------------------------

    st.subheader("Borrower Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Loan Amount**")
        st.write(f"${loan_amnt:,.0f}")

        st.write("**Interest Rate**")
        st.write(f"{int_rate:.2f}%")

        st.write("**Debt-to-Income Ratio**")
        st.write(f"{dti:.2f}")

        st.write("**Revolving Utilisation**")
        st.write(f"{revol_util:.2f}%")

    with col2:

        st.write("**Annual Income**")
        st.write(f"${annual_inc:,.0f}")

        st.write("**Loan Grade**")
        st.write(grade)

        st.write("**Loan Term**")
        st.write(term)

        st.write("**Home Ownership**")
        st.write(home_ownership)

    st.divider()


# Explainable AI Results
# __________________________________________________________

if prediction is not None and explanation is not None:

    st.header("Explainable AI")

    
    # SHAP Feature Contributions
    # ------------------------------------------

    shap_df = shap_bar_chart(explanation)

    st.divider()

    
    # AI Decision Summary
    # ------------------------------------------

    display_summary(prediction,shap_df)

    st.divider()

    
    # Feature Importance Table
    # ------------------------------------------

    st.subheader("Top Features Driving the Prediction")

    display_df = shap_df.copy()

    # Only keep the two columns needed for display

    display_df = display_df[["feature", "shap_value"]]

    display_df = display_df.rename(

        columns={

            "feature": "Feature",

            "shap_value": "SHAP Value"})

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True)

    
    # Prediction Report
    # ------------------------------------------

    report = f"""

Explainable AI Credit Risk Decision Report
_____________________________________________________

Prediction
----------
Risk Class:
{prediction['risk_class']}

Probability of Default:
{prediction['default_probability']:.2%}

Model Confidence:
{prediction['confidence']:.2%}

----------------------------------------------------

Applicant Information

Loan Amount: ${loan_amnt:,.0f}
Annual Income: ${annual_inc:,.0f}
Interest Rate: {int_rate:.2f}%
Debt-to-Income Ratio: {dti:.2f}
Revolving Utilisation: {revol_util:.2f}%
Loan Grade: {grade}
Loan Term: {term}
Home Ownership: {home_ownership}

----------------------------------------------------

Top Features

    {display_df.to_string(index=False)}


Generated by Explainable AI Credit Risk Platform
______________________________________________________
"""

    pdf = create_pdf(prediction,application,shap_df)

    with open(pdf, "rb") as f:

        st.download_button(
            
            "📄 Download Credit Decision Report",
            
            f,
            
            file_name="Credit_Decision_Report.pdf",
            
            mime="application/pdf")


# Footer
# ______________________________________________________

st.markdown(
"""
---
**Explainable AI Credit Risk Decision Platform**

Built using:

- XGBoost
- SHAP Explainability
- FastAPI
- Streamlit
- Python

""")
