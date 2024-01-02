# utils/studio_style.py

import streamlit as st

def keyword_label(text):
    return f'<div class="keyword-label">{text}</div>'

def apply_studio_style():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300&display=swap');

            body {
                font-family: 'Open Sans', sans-serif;
                margin: 0;
                padding: 0;
                color: #333;
                background-color: #f7f7f7;
            }

            .keyword-label {
                background-color: RGB(67, 4, 77); /* Blue background color */
                color: #fff; /* White text color */
                border-radius: 5px;
                padding: 5px 10px;
                display: inline-block;
                margin-right: 10px;
                margin-top: 10px;
                font-size: 15px;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )
