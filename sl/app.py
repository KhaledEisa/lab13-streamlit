#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:52:09 2025

@author: noob
"""

# !streamlit run app.py --server.headless true

import streamlit as st
import joblib
import numpy as np

model = BASE_DIR / "modeling" / "svc_model.pkl"


label_encoder = BASE_DIR / "modeling" / "label_encoder.pkl"
questions = BASE_DIR / "modeling" / "questions.pkl"

st.title("🧠 Personality Prediction App (Lab13)")

st.markdown("Rate the following statements on a scale from **Fully Disagree (-3)** to **Fully Agree (3)**.")

likert_options = {
    "Fully Agree": 3,
    "Partially Agree": 2,
    "Slightly Agree": 1,
    "Neutral": 0,
    "Slightly Disagree": -1,
    "Partially Disagree": -2,
    "Fully Disagree": -3
}

answers = []
unanswered = False

st.write("Please respond to **all** questions before submitting.")

for question in questions:
    response = st.radio(
        f"{question}?",
        options=[None] + list(likert_options.keys()),
        key=question,
        index=0,  # None is default
        format_func=lambda x: "Select an option" if x is None else x
    )
    if response is None:
        unanswered = True
    else:
        answers.append(likert_options[response])

if st.button("Submit"):
    if unanswered or len(answers) < len(questions):
        st.error("❗ Please answer all the questions before submitting.")
    else:
        input_data = np.array(answers).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        persona = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🎯 Predicted Personality: **{persona}**")
