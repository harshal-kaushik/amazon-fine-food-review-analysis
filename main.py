import streamlit as st
import pickle
import re

# ===============================
# PAGE CONFIG (MUST BE FIRST)
# ===============================
st.set_page_config(
    page_title="Amazon Dine Review Intelligence",
    page_icon="🍽️",
    layout="wide"
)

# ===============================
# CUSTOM CSS (🔥 UI MAGIC)
# ===============================
st.markdown("""
<style>

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #FF4B4B, #FF9F1C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

/* Result card */
.result-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    text-align: center;
}

/* Sentiment badge */
.badge-positive {
    background-color: #e6f4ea;
    color: #1e7e34;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
}

.badge-negative {
    background-color: #fdecea;
    color: #c62828;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# MODEL SETTINGS
# ===============================
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
# TOPIC DETECTION
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

# ===============================
# HEADER UI
# ===============================
st.markdown('<div class="main-title">🍽️ Amazon Dine Review Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analyze customer reviews using NLP to detect sentiment and key issues</div>',
    unsafe_allow_html=True
)

# ===============================
# INPUT
# ===============================
review = st.text_area(
    "✍️ Enter customer review:",
    height=180,
    placeholder="Example: Food was cold and delivery was very late..."
)

analyze_btn = st.button("🔍 Analyze Review", use_container_width=True)

# ===============================
# PREDICTION
# ===============================
if analyze_btn:

    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
        st.stop()

    clean_review = clean_text(review)
    vectorized_review = vectorizer.transform([clean_review])
    prob = model.predict_proba(vectorized_review)[0, 1]
    pred = int(prob >= BEST_THRESHOLD)

    sentiment_label = "Positive 😊" if pred == 0 else "Negative 😞"
    topics = detect_topics(clean_review)

    ## RESULTS
    st.markdown("### 📊 Result")

    badge_class = "badge-positive" if pred == 0 else "badge-negative"

    st.markdown(
        f'<span class="{badge_class}">{sentiment_label}</span>',
        unsafe_allow_html=True
    )

    st.caption(f"Confidence: {prob:.2f}")

    st.markdown("### 🔎 Detected Issues")

    if topics:
        for topic in topics:
            st.write(f"• {topic}")
    else:
        st.write("✅ No major issues detected")