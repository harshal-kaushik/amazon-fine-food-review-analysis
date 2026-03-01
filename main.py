import streamlit as st
import pickle
import re

# best threshold
BEST_THRESHOLD = 0.70

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

@st.cache_resource
def load_artifacts():
    model = pickle.load(open("models/sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# ===============================
# TOPIC / ISSUE DETECTION
# ===============================
TOPIC_KEYWORDS = {
    "Delivery Issue": ["late", "delay", "slow delivery", "not delivered"],
    "Food Quality": ["cold", "stale", "tasteless", "bad taste"],
    "Packaging Issue": ["leak", "spill", "damaged", "packaging"],
    "Missing Items": ["missing", "not received", "incomplete"],
}

def detect_topics(text):
    text_lower = text.lower()
    found = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(word in text_lower for word in keywords):
            found.append(topic)

    return found

# ==================
# UI
# ==================
st.set_page_config(
    page_title="Amazon Dine Review Intelligence",
    page_icon="🍽️",
    layout="centered"
)

st.title("Amazon Dine Review Intelligence")
st.markdown(
    "Analyze customer reviews using NLP to detect sentiment and key issues."
)

review = st.text_area(
    "✍️ Enter customer review:",
    height=150,
    placeholder="Example: Food was cold and delivery was very late..."
)

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    # clean
    clean_review = clean_text(review)

    # vectorize
    vectorized_review = vectorizer.transform([clean_review])

    # probability
    prob = model.predict_proba(vectorized_review)[0, 1]

    # threshold decision
    pred = int(prob >= BEST_THRESHOLD)

    sentiment_label = "Positive 😊" if pred == 0 else "Negative 😞"

    # topic
    topics = detect_topics(clean_review)



## RESULTS
    st.subheader("Analyzed Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentiment", sentiment_label)

    with col2:
        st.metric("Confidence", f"{prob:.2f}")

    st.markdown("### 🔎 Detected Issues")

    if topics:
        for topic in topics:
            st.write(f"• {topic}")
    else:
        st.write("✅ No major issues detected")




