"""
Professional PDF Report Generator
"""

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle)

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib import colors


def create_pdf(
    prediction,
    application,
    shap_df,
    filename="credit_report.pdf"
):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(filename)

    story = []

    story.append(
        Paragraph(
            "<b>Explainable AI Credit Risk Decision Report</b>",
            styles["Title"]))

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            f"<b>Risk Class:</b> {prediction['risk_class']}",
            styles["BodyText"]))

    story.append(
        Paragraph(
            f"<b>Probability of Default:</b> {prediction['default_probability']:.2%}",
            styles["BodyText"]))

    story.append(
        Paragraph(
            f"<b>Model Confidence:</b> {prediction['confidence']:.2%}",
            styles["BodyText"]))

    story.append(Spacer(1,20))

    borrower = [

        ["Feature","Value"],

        ["Loan Amount",application["loan_amnt"]],

        ["Annual Income",application["annual_inc"]],

        ["Interest Rate",application["int_rate"]],

        ["DTI",application["dti"]],

        ["Grade",application["grade"]],

        ["Home Ownership",application["home_ownership"]]]

    table = Table(borrower)

    table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.darkblue),

            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

            ("GRID",(0,0),(-1,-1),1,colors.grey),

            ("BACKGROUND",(0,1),(-1,-1),colors.beige)]))

    story.append(table)

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            "<b>Top SHAP Features</b>",
            styles["Heading2"]))

    for _, row in shap_df.head(10).iterrows():

        story.append(

            Paragraph(

                f"{row['feature']} : {row['shap_value']:.3f}",

                styles["BodyText"]))

    doc.build(story)

    return filename