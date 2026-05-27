# app.py
# ============================================
# Mental Health Sentiment Monitoring System
# Streamlit Deployment
# ============================================

import streamlit as st
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================
# Load Saved Files
# ============================================

model = load_model("mental_health_rnn_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

encoder = joblib.load("label_encoder.pkl")

# ============================================
# App Configuration
# ============================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="centered"
)

# ============================================
# Title
# ============================================

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.write("""
This application predicts emotional sentiment
from user text messages using a Simple RNN model.
""")

# ============================================
# User Input
# ============================================

user_input = st.text_area(
    "Enter your message:",
    height=150
)

# ============================================
# Prediction Function
# ============================================

max_length = 50

def predict_emotion(text):

    # Lowercase conversion
    text = text.lower()

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    # Prediction
    prediction = model.predict(padded, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence_score = np.max(prediction)

    emotion = encoder.inverse_transform([predicted_class])[0]

    return emotion, confidence_score

# ============================================
# Predict Button
# ============================================

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")

    else:

        emotion, confidence = predict_emotion(user_input)

        st.success("Prediction Completed")

        # Display Emotion
        st.subheader("Predicted Emotion")
        st.write(emotion)

        # Display Confidence
        st.subheader("Confidence Score")
        st.write(f"{round(confidence * 100, 2)} %")

        # Emotional Feedback
        st.subheader("Emotional Status Feedback")

        if confidence > 0.80:
            st.error("Strong emotional indication detected.")

        elif confidence > 0.60:
            st.warning("Moderate emotional indication detected.")

        else:
            st.info("Low confidence prediction.")

# ============================================
# Footer
# ============================================

st.markdown("---")
st.write("Developed using Streamlit + TensorFlow + Simple RNN")