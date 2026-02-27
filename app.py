import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime

@st.cache_resource
def load_model():
    data = joblib.load('model.pkl')
    return data['model'], data['vectorizer']

@st.cache_data
def load_training_data():
    data = pd.read_csv(
        '/Users/berkebarantozkoparan/Desktop/spam detection/spam.csv',
        encoding='latin-1'
    )
    data['processed_text'] = data['Message'].apply(preprocess_text)
    return data

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text, model, vectorizer):
    processed = preprocess_text(text)
    tfidf = vectorizer.transform([processed])
    label = model.predict(tfidf)[0]
    proba = model.predict_proba(tfidf)[0]
    confidence = proba[label] * 100
    return label, confidence

def make_wordcloud(text, colormap):
    wc = WordCloud(width=700, height=350, background_color='black',
                   colormap=colormap, max_words=80).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('black')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="🔍", layout="wide")

# Load model and data
model, vectorizer = load_model()
train_data = load_training_data()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "spam_count" not in st.session_state:
    st.session_state.spam_count = 0
if "ham_count" not in st.session_state:
    st.session_state.ham_count = 0
if "history" not in st.session_state:
    st.session_state.history = []  # list of {time, label, message}

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Spam Detector")
    st.markdown("---")

    st.subheader("📊 Statistics")
    total = st.session_state.spam_count + st.session_state.ham_count
    col1, col2 = st.columns(2)
    col1.metric("Spam", st.session_state.spam_count)
    col2.metric("Ham", st.session_state.ham_count)

    if total > 0:
        spam_pct = st.session_state.spam_count / total * 100
        st.progress(int(spam_pct), text=f"Spam rate: {spam_pct:.0f}%")

    st.markdown("---")
    st.subheader("🤖 Model Info")
    st.markdown("""
- **Model:** Logistic Regression
- **Features:** TF-IDF (5000 tokens)
- **Accuracy:** 96.77%
- **Dataset:** SMS Spam Collection
""")
    st.markdown("---")
    st.subheader("📖 How It Works")
    st.markdown("""
1. Message is cleaned (lowercase, punctuation removed)
2. Converted to a numeric vector via TF-IDF
3. Model predicts whether it is spam or not
4. Result is shown with a confidence score
""")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.spam_count = 0
        st.session_state.ham_count = 0
        st.session_state.history = []
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "☁️ Word Cloud", "📈 History"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────
with tab1:
    st.header("💬 Message Analysis")
    st.caption("Type a message and instantly find out if it's spam or ham.")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                label = msg["label"]
                confidence = msg["confidence"]
                if label == 1:
                    st.markdown(f"""
<div style="background:#4a1010;border-left:5px solid #ff4b4b;padding:14px 18px;border-radius:8px;">
  <span style="font-size:22px;">⚠️</span>
  <strong style="font-size:18px;color:#ff4b4b;margin-left:8px;">SPAM</strong>
  <div style="margin-top:6px;color:#ffaaaa;">This message is likely spam.</div>
  <div style="margin-top:8px;">
    <strong style="color:#fff;">Confidence:</strong>
    <span style="color:#ff4b4b;font-size:20px;margin-left:8px;">{confidence:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background:#0f3d1f;border-left:5px solid #21c55d;padding:14px 18px;border-radius:8px;">
  <span style="font-size:22px;">✅</span>
  <strong style="font-size:18px;color:#21c55d;margin-left:8px;">HAM</strong>
  <div style="margin-top:6px;color:#aaffcc;">This message looks legitimate.</div>
  <div style="margin-top:8px;">
    <strong style="color:#fff;">Confidence:</strong>
    <span style="color:#21c55d;font-size:20px;margin-left:8px;">{confidence:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        label, confidence = predict(prompt, model, vectorizer)

        if label == 1:
            st.session_state.spam_count += 1
        else:
            st.session_state.ham_count += 1

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "label": "Spam" if label == 1 else "Ham",
            "message": prompt[:60] + ("..." if len(prompt) > 60 else ""),
            "confidence": confidence,
        })

        st.session_state.messages.append({"role": "assistant", "label": label, "confidence": confidence})
        st.rerun()

# ── Tab 2: Word Cloud ─────────────────────────────────────────────────────
with tab2:
    st.header("☁️ Word Cloud")
    st.caption("Most frequent words found in spam and ham messages from the training dataset.")

    spam_text = " ".join(train_data[train_data['Category'] == 'spam']['processed_text'])
    ham_text  = " ".join(train_data[train_data['Category'] == 'ham']['processed_text'])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Spam Words")
        st.pyplot(make_wordcloud(spam_text, "Reds"))
    with col2:
        st.subheader("✅ Ham Words")
        st.pyplot(make_wordcloud(ham_text, "Greens"))

# ── Tab 3: History ────────────────────────────────────────────────────────
with tab3:
    st.header("📈 Session History")

    if not st.session_state.history:
        st.info("No messages analysed yet. Go to the Chat tab and send some messages.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # Bar chart
        counts = df['label'].value_counts().reindex(["Spam", "Ham"], fill_value=0)
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        bars = ax.bar(counts.index, counts.values,
                      color=["#ff4b4b", "#21c55d"], width=0.4)
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(val), ha='center', color='white', fontsize=12)
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Message Log")
        st.dataframe(
            df.rename(columns={
                "time": "Time",
                "label": "Result",
                "message": "Message",
                "confidence": "Confidence (%)"
            }),
            use_container_width=True,
            hide_index=True,
        )
