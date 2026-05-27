import streamlit as st
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("mental_health_rnn_model.h5")

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="centered"
)

# ============================================
# LOAD MODEL FILES
# ============================================

model = load_model("mental_health_rnn_model.keras")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

encoder = joblib.load("label_encoder.pkl")

# ============================================
# PARAMETERS
# ============================================

max_length = 50

# ============================================
# HEADER
# ============================================

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

st.markdown("---")

# ============================================
# ABOUT PROJECT
# ============================================

st.header("📘 About the Project")

st.write("""
Emotional AI helps computers understand human emotions
through Natural Language Processing (NLP).

This project uses a Simple Recurrent Neural Network (RNN)
to analyze emotional sentiment from text messages.

Applications include:
- mental wellness monitoring
- emotional trend analysis
- AI counseling support
- early emotional intervention

RNN models are powerful for sequence learning because
they remember previous words while processing text.
""")

st.markdown("---")

# ============================================
# USER INPUT AREA
# ============================================

st.header("✍ User Text Input")

st.write("Sample Sentences:")

st.code("I feel very stressed and anxious")
st.code("Today was a beautiful and happy day")
st.code("Nobody understands my feelings")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=150
)

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_emotion(text):

    text = text.lower()

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(padded, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    emotion = encoder.inverse_transform([predicted_index])[0]

    return emotion, confidence, prediction[0]

# ============================================
# PREDICTION BUTTON
# ============================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        emotion, confidence, probs = predict_emotion(user_input)

        st.markdown("---")

        # ====================================
        # PREDICTION OUTPUT
        # ====================================

        st.header("📊 Prediction Output")

        st.success(f"Emotion Detected: {emotion}")

        st.info(f"Confidence Score: {round(confidence * 100, 2)}%")

        # Emotional status
        if confidence > 0.80:
            status = "Strong emotional indication detected"
        elif confidence > 0.60:
            status = "Moderate emotional indication detected"
        else:
            status = "Low confidence emotional prediction"

        st.warning(f"Emotional Status: {status}")

        # ====================================
        # VISUALIZATION AREA
        # ====================================

        st.header("📈 Sentiment Confidence Graph")

        class_labels = encoder.classes_

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(class_labels, probs)

        ax.set_xlabel("Emotion Categories")
        ax.set_ylabel("Probability")

        ax.set_title("Emotion Probability Distribution")

        st.pyplot(fig)

        # ====================================
        # EMOTIONAL GUIDANCE AREA
        # ====================================

        st.header("💡 Emotional Wellness Guidance")

        emotion_lower = emotion.lower()

        if "anxiety" in emotion_lower:

            st.write("✔ Take a short break")
            st.write("✔ Practice deep breathing")
            st.write("✔ Try calming music")

        elif "depression" in emotion_lower:

            st.write("✔ Talk to someone you trust")
            st.write("✔ Spend time outdoors")
            st.write("✔ Maintain healthy sleep")

        elif "stress" in emotion_lower:

            st.write("✔ Try meditation")
            st.write("✔ Organize your tasks")
            st.write("✔ Take regular breaks")

        elif "happy" in emotion_lower or "normal" in emotion_lower:

            st.write("✔ Keep maintaining positivity")
            st.write("✔ Continue healthy habits")
            st.write("✔ Share positivity with others")

        else:

            st.write("✔ Stay positive")
            st.write("✔ Practice mindfulness")
            st.write("✔ Take care of your mental wellness")

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.write(
    "Developed using Streamlit, TensorFlow, NLP, and Simple RNN"
)
