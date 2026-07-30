"""
=========================================================
Risk Gauge Component
=========================================================
"""

import plotly.graph_objects as go

def risk_gauge(default_probability, risk_class):
    """
    Returns a Plotly gauge chart.
    """

    probability = float(default_probability) * 100

    if risk_class == "Low Risk":
        bar_color = "green"
    else:
        bar_color = "red"

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability,

            number={

                "suffix": "%",

                "font": {

                    "size": 40}},

            title={

                "text": "<b>Probability of Default</b>",

                "font": {"size": 22}},

            gauge={

                "axis": {

                    "range": [0, 100]},

                "bar": {

                    "color": bar_color},

                "steps": [

                    {

                        "range": [0, 30],

                        "color": "#7CFC00"

                    },

                    {

                        "range": [30, 50],

                        "color": "#FFD700"

                    },

                    {

                        "range": [50, 70],

                        "color": "#FFA500"

                    },

                    {

                        "range": [70, 100],

                        "color": "#FF4B4B"

                    }],

                "threshold": {

                    "line": {

                        "color": "black",

                        "width": 6},

                    "value": probability}}))

    fig.update_layout(

    paper_bgcolor="white",

    plot_bgcolor="white",

    font=dict(

        color="#222",

        size=18),

    margin=dict(

        l=20,

        r=20,

        t=80,

        b=20),

    height=350)

    return fig