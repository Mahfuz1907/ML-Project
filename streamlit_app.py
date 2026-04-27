import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

# PAGE CONFIG
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# LOAD MODEL
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# PREPROCESSING (must match train_model.py)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def predict_spam(text, model, vectorizer):
    cleaned = preprocess(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    # Get probability if model supports it
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        spam_prob = proba[1]
    else:
        # For LinearSVC, use decision function
        decision = model.decision_function(features)[0]
        # Convert to a rough probability using sigmoid
        import math
        spam_prob = 1 / (1 + math.exp(-decision))
    return prediction, spam_prob

# SPAM KEYWORD HIGHLIGHTER
SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'prize', 'cash', 'claim',
    'urgent', 'congratulations', 'call now', 'click', 'offer',
    'guaranteed', 'selected', 'exclusive', 'reward', 'bonus',
    'credit', 'loan', 'discount', 'deal', 'limited', 'expires'
]

def highlight_keywords(text):
    highlighted = text
    for kw in SPAM_KEYWORDS:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        highlighted = pattern.sub(f'<mark style="background:#FAEEDA;color:#633806;padding:1px 3px;border-radius:3px;">{kw.upper()}</mark>', highlighted)
    return highlighted

# CUSTOM CSS
st.markdown("""
<style>
    .main { max-width: 720px; margin: auto; }
    .result-spam {
        background: #FCEBEB; border: 1px solid #F09595;
        border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .result-ham {
        background: #EAF3DE; border: 1px solid #97C459;
        border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .result-title { font-size: 22px; font-weight: 600; margin-bottom: 0.3rem; }
    .result-subtitle { font-size: 14px; margin-bottom: 0.8rem; }
    .metric-row { display: flex; gap: 12px; margin: 1rem 0; flex-wrap: wrap; }
    .metric-box {
        background: #F7F7F5; border-radius: 8px; padding: 10px 16px;
        flex: 1; min-width: 100px; text-align: center;
    }
    .metric-val { font-size: 20px; font-weight: 600; }
    .metric-lbl { font-size: 11px; color: #666; margin-top: 2px; }
    .tip-box {
        background: #E6F1FB; border-left: 3px solid #185FA5;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        font-size: 13px; margin-top: 1rem;
    }
    .example-btn { margin: 4px; }
</style>
""", unsafe_allow_html=True)

# MAIN APP
st.title("🛡️ SMS & Email Spam Detector")
st.markdown("Classify messages as **Spam** or **Ham** using Machine Learning (Naive Bayes + TF-IDF)")
st.divider()

# Load model
if not os.path.exists('model.pkl'):
    st.error("⚠️ Model not found! Please run `python train_model.py` first.")
    st.stop()

model, vectorizer = load_model()


# Text input
default_text = st.session_state.get('input_text', '')
user_input = st.text_area(
    "Enter your message here:",
    value=default_text,
    height=130,
    placeholder="Type or paste any SMS or email message...",
    key="message_input"
)

# Predict button
predict_clicked = st.button("Detect Spam", type="primary", use_container_width=True)

if predict_clicked and user_input.strip():
    prediction, spam_prob = predict_spam(user_input, model, vectorizer)
    ham_prob = 1 - spam_prob
    word_count = len(user_input.split())
    char_count = len(user_input)

    if prediction == 1:
        # SPAM result
        st.markdown(f"""
        <div class="result-spam">
            <div class="result-title" style="color:#A32D2D;">🚨 SPAM DETECTED</div>
            <div class="result-subtitle" style="color:#791F1F;">
                This message has characteristics of spam.
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val" style="color:#A32D2D;">{spam_prob*100:.1f}%</div>
                    <div class="metric-lbl">Spam confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{word_count}</div>
                    <div class="metric-lbl">Words</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{char_count}</div>
                    <div class="metric-lbl">Characters</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # HAM result
        st.markdown(f"""
        <div class="result-ham">
            <div class="result-title" style="color:#27500A;">✅ NOT SPAM (Ham)</div>
            <div class="result-subtitle" style="color:#3B6D11;">
                This message appears to be legitimate.
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val" style="color:#27500A;">{ham_prob*100:.1f}%</div>
                    <div class="metric-lbl">Ham confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{word_count}</div>
                    <div class="metric-lbl">Words</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{char_count}</div>
                    <div class="metric-lbl">Characters</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    st.markdown("**Probability breakdown:**")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("Ham probability", f"{ham_prob*100:.1f}%")
        st.progress(float(ham_prob))
    with prob_col2:
        st.metric("Spam probability", f"{spam_prob*100:.1f}%")
        st.progress(float(spam_prob))

    # Keyword highlighter
    st.markdown("**Keyword analysis:**")
    highlighted = highlight_keywords(user_input)
    st.markdown(
        f'<div style="background:#F7F7F5;padding:12px 16px;border-radius:8px;line-height:1.8;">{highlighted}</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:
        found_kw = [kw for kw in SPAM_KEYWORDS if kw.lower() in user_input.lower()]
        if found_kw:
            st.markdown(
                f'<div class="tip-box">Spam keywords detected: <b>{", ".join(found_kw)}</b></div>',
                unsafe_allow_html=True
            )

elif predict_clicked and not user_input.strip():
    st.warning("Please enter a message to analyze.")
