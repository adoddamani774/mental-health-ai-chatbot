import json
import os
import random
import streamlit as st
from textblob import TextBlob
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="Mental Health AI Chatbot",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .safety-banner {
        background-color: #e2f0d9;
        border-left: 5px solid #548235;
        padding: 10px 15px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #274e13;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Intents File with Error Handling
@st.cache_data
def load_intents():
    if not os.path.exists("intents.json"):
        return {"intents": []}
    try:
        with open("intents.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading intents.json: {e}")
        return {"intents": []}

intents_data = load_intents()

# Load Hugging Face Emotion Model with Error Handling
@st.cache_resource
def load_emotion_pipeline():
    try:
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        return classifier
    except Exception as e:
        return None

emotion_classifier = load_emotion_pipeline()

# Helper Function for Intent Matching
def get_intent_response(user_input):
    user_input_lower = user_input.lower()
    for intent in intents_data.get("intents", []):
        for pattern in intent.get("patterns", []):
            if pattern in user_input_lower:
                return random.choice(intent.get("responses", ["I hear you. Tell me more."]))
    return None

# Helper Function for Sentiment Analysis (TextBlob)
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive 😊"
    elif polarity < -0.1:
        return "Negative 😔"
    else:
        return "Neutral 😐"

# Helper Function for Emotion Detection
def detect_emotion(text):
    if emotion_classifier is None:
        return "Model unavailable"
    try:
        results = emotion_classifier(text)
        if results and len(results) > 0:
            top_result = results[0]
            emotion = top_result['label'].capitalize()
            score = top_result['score']
            return f"{emotion} ({score:.2f})"
    except Exception:
        return "Analysis failed"
    return "Unknown"

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/mental-health.png", width=80)
    st.title("Wellbeing Companion")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write(
        "This chatbot provides a safe space for general emotional support, "
        "mindfulness, and reflection using AI-powered sentiment and emotion analysis."
    )
    st.markdown("---")
    
    # Emergency Resources
    st.markdown("### 🆘 Emergency Helplines")
    st.markdown("""
    - **US:** Call or text `988`
    - **UK:** Call `111` or text `85258`
    - **India:** Call `9152987821` (Vandrevala Foundation)
    - **International:** [Befrienders Worldwide](https://www.befrienders.org/)
    """)
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Application Interface
st.title("🧠 Mental Health AI Chatbot")
st.write("A supportive space for your thoughts, feelings, and reflection.")

# Safety Disclaimer Banner
st.markdown("""
<div class="safety-banner">
    <b>⚠️ Important Safety Notice:</b> This chatbot is designed for general emotional support and friendly conversation only. 
    It is <b>not</b> a medical device, does <b>not</b> diagnose mental health conditions, and cannot replace professional medical help or therapy. 
    If you are in crisis, please reach out to local emergency services or a healthcare professional immediately.
</div>
""", unsafe_allow_html=True)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your wellbeing companion. How are you feeling today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if user_prompt := st.chat_input("Type your thoughts or feelings here..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Perform Sentiment & Emotion Analysis
    sentiment = analyze_sentiment(user_prompt)
    emotion = detect_emotion(user_prompt)

    # Generate Response (Intent Matching or Fallback)
    bot_response = get_intent_response(user_prompt)
    if not bot_response:
        # Fallback response based on sentiment
        if "Negative" in sentiment:
            bot_response = "I hear that you're going through a challenging moment right now. Would you like to talk about what's weighing on your mind?"
        elif "Positive" in sentiment:
            bot_response = "It's wonderful to hear some positivity in your words! What's making you feel good today?"
        else:
            bot_response = "Thank you for sharing that with me. I'm listening—feel free to share more whenever you're ready."

    # Append Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        st.caption(f"📊 Sentiment: {sentiment} | 🎭 Detected Emotion: {emotion}")