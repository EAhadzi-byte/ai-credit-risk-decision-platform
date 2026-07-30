"""
=========================================================
AI Decision Summary
=========================================================
"""

import streamlit as st


def display_summary(prediction, shap_df):

    probability = prediction["default_probability"]

    confidence = prediction["confidence"]

    risk = prediction["risk_class"]

    if shap_df.empty:

        st.info("No explanation available.")

        return

    positive = shap_df[

        shap_df["shap_value"] > 0]

    negative = shap_df[

        shap_df["shap_value"] < 0]

    st.subheader("AI Decision Summary")

    if risk == "Low Risk":

        st.success(

f"""
### Lending Recommendation

The applicant has been classified as **LOW RISK**.

Estimated probability of default:

**{probability:.2%}**

Model confidence:

**{confidence:.2%}**

The model identified several characteristics associated with lower credit risk.

This application is suitable for standard lending policy, subject to affordability checks.
""")

    else:

        st.error(

f"""
### Lending Recommendation

The applicant has been classified as **HIGH RISK**.

Estimated probability of default:

**{probability:.2%}**

Model confidence:

**{confidence:.2%}**

Several borrower characteristics increase the estimated probability of default.

The application should be reviewed manually before approval.
"""
        )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Factors Increasing Risk")

        if positive.empty:

            st.write("None")

        else:

            for _, row in positive.head(5).iterrows():

                st.write(

                    f"• **{row['feature']}** (+{row['shap_value']:.3f})")

    with col2:

        st.markdown("### Factors Reducing Risk")

        if negative.empty:

            st.write("None")

        else:

            for _, row in negative.head(5).iterrows():

                st.write(

                    f"• **{row['feature']}** ({row['shap_value']:.3f})")