import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr
import io

# ---------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------
st.set_page_config(page_title="🧠 Emotion Detector", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: #F5F5F5;
}
h1, h2, h3 {
    color: #FF6EC7;
}
.stButton>button {
    background-color: #FF6EC7;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}
.emotion-box {
    background-color: #161A22;
    border-left: 4px solid #FF6EC7;
    border-radius: 8px;
    padding: 16px;
    margin-top: 10px;
}
.highlight-box {
    background-color: #161A22;
    border-radius: 8px;
    padding: 16px;
    margin-top: 10px;
    line-height: 2.2;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

EMOTION_EMOJIS = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲"
}

EMOTION_TRANSLATIONS = {
    "joy":      {"en": "joy",      "hi": "खुशी",       "bn": "আনন্দ"},
    "sadness":  {"en": "sadness",  "hi": "उदासी",      "bn": "দুঃখ"},
    "anger":    {"en": "anger",    "hi": "गुस्सा",      "bn": "রাগ"},
    "fear":     {"en": "fear",     "hi": "डर",         "bn": "ভয়"},
    "love":     {"en": "love",     "hi": "प्यार",      "bn": "ভালোবাসা"},
    "surprise": {"en": "surprise", "hi": "आश्चर्य",     "bn": "বিস্ময়"},
}

SPEECH_TEMPLATE = {
    "en": "The detected emotion is {emotion}, with {confidence} percent confidence.",
    "hi": "पहचानी गई भावना {emotion} है, {confidence} प्रतिशत विश्वास के साथ।",
    "bn": "সনাক্ত করা আবেগ হলো {emotion}, {confidence} শতাংশ আত্মবিশ্বাসের সাথে।",
}

LANGUAGE_OPTIONS = {"English": "en", "Hindi": "hi", "Bengali": "bn"}


# ---------------------------------------------------------------
# LOAD MODEL (cached so it only loads once, not on every interaction)
# ---------------------------------------------------------------
@st.cache_resource
def load_emotion_classifier():
    """Load the pretrained DistilBERT emotion classification model.
    This downloads automatically (~260MB) the first time it runs."""
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None
    )


def detect_emotion(text, classifier):
    """Run the text through the model and return sorted emotion scores"""
    results = classifier(text)[0]
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted


def generate_voice(emotion_label, confidence, lang_code):
    """Build a spoken sentence in the chosen language and return MP3 bytes."""
    emotion_word = EMOTION_TRANSLATIONS.get(emotion_label, {}).get(lang_code, emotion_label)
    sentence = SPEECH_TEMPLATE[lang_code].format(
        emotion=emotion_word,
        confidence=f"{confidence:.1f}"
    )
    tts = gTTS(text=sentence, lang=lang_code)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


def transcribe_audio(audio_file):
    """Convert a recorded audio clip into text using Google's speech recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)


def get_word_importance(text, classifier, target_label, base_score):
    """
    Occlusion-based explainability: remove one word at a time and see how
    much the target emotion's confidence drops. Bigger drop = more important word.
    Capped at 25 words to keep it fast.
    """
    words = text.split()
    importances = []
    for i in range(len(words)):
        modified_text = " ".join(words[:i] + words[i+1:])
        if not modified_text.strip():
            importances.append(0)
            continue
        result = classifier(modified_text)[0]
        score_dict = {r["label"]: r["score"] for r in result}
        modified_score = score_dict.get(target_label, 0)
        importances.append(max(base_score - modified_score, 0))

    max_imp = max(importances) if importances and max(importances) > 0 else 1
    normalized = [imp / max_imp for imp in importances]
    return words, normalized


def render_highlighted_text(words, importances):
    """Return HTML where each word's background opacity reflects its importance."""
    spans = []
    for word, imp in zip(words, importances):
        opacity = 0.15 + (imp * 0.75)  # keep a visible floor even for low-importance words
        spans.append(
            f'<span style="background-color: rgba(255,110,199,{opacity:.2f}); '
            f'padding:2px 5px; border-radius:5px; margin:2px; display:inline-block;">{word}</span>'
        )
    return " ".join(spans)


# ---------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "spoken_text" not in st.session_state:
    st.session_state.spoken_text = ""

# ---------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------
st.title("🧠 Emotion Detector")
st.markdown("Type how you're feeling, or paste any text — a real deep learning model (DistilBERT) analyzes the emotional tone.")

with st.spinner("Loading AI model (first time only, ~260MB download)..."):
    classifier = load_emotion_classifier()

tab1, tab2 = st.tabs(["🔍 Detect Emotion", "📊 Mood History"])

# =================================================================
# TAB 1: DETECT EMOTION
# =================================================================
with tab1:
    # --- Voice input (speech-to-text) ---
    st.markdown("**🎤 Or speak how you're feeling:**")
    audio_value = st.audio_input("Click to record")

    if audio_value is not None:
        with st.spinner("Transcribing your voice..."):
            try:
                st.session_state.spoken_text = transcribe_audio(audio_value)
                st.success(f"Heard: \"{st.session_state.spoken_text}\"")
            except sr.UnknownValueError:
                st.error("Sorry, couldn't understand the audio. Please try again.")
            except sr.RequestError:
                st.error("Speech recognition service unavailable. Check your internet connection.")

    text_input = st.text_area(
        "What's on your mind?",
        value=st.session_state.spoken_text,
        placeholder="e.g. I just got the internship I was hoping for, I can't believe it!",
        height=120
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        voice_lang_label = st.selectbox("🔊 Voice language for result", list(LANGUAGE_OPTIONS.keys()))
        voice_lang_code = LANGUAGE_OPTIONS[voice_lang_label]
    with col_b:
        show_explain = st.checkbox("🔬 Show word-level explanation", value=True)

    if st.button("🔍 Analyze Emotion"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                results = detect_emotion(text_input, classifier)
                top_emotion = results[0]

            emoji = EMOTION_EMOJIS.get(top_emotion["label"], "🙂")
            st.markdown(f"""
            <div class="emotion-box">
            <h2>{emoji} {top_emotion['label'].capitalize()}</h2>
            <p>Confidence: {top_emotion['score']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Voice output ---
            with st.spinner("Generating voice..."):
                try:
                    audio = generate_voice(
                        top_emotion["label"],
                        top_emotion["score"] * 100,
                        voice_lang_code
                    )
                    st.audio(audio, format="audio/mp3")
                except Exception as e:
                    st.info(f"Voice generation needs an internet connection (gTTS). ({e})")

            # --- Word-level explainability ---
            if show_explain:
                word_count = len(text_input.split())
                if word_count <= 25:
                    with st.spinner("Figuring out which words mattered..."):
                        words, importances = get_word_importance(
                            text_input, classifier, top_emotion["label"], top_emotion["score"]
                        )
                    st.markdown("**🔬 Why this emotion? (darker = more influential word)**")
                    st.markdown(
                        f'<div class="highlight-box">{render_highlighted_text(words, importances)}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Word-level explanation works best under 25 words — try a shorter sentence to see it.")

            fig = go.Figure(go.Bar(
                x=[r["score"] * 100 for r in results],
                y=[f"{EMOTION_EMOJIS.get(r['label'], '')} {r['label'].capitalize()}" for r in results],
                orientation="h",
                marker=dict(color="#FF6EC7")
            ))
            fig.update_layout(
                title="Emotion Breakdown",
                xaxis_title="Confidence (%)",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#161A22",
                font=dict(color="white"),
                height=350
            )
            st.plotly_chart(fig, width="stretch")

            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Text": text_input[:100] + ("..." if len(text_input) > 100 else ""),
                "Emotion": top_emotion["label"],
                "Confidence": round(top_emotion["score"] * 100, 1)
            })

            # Clear the spoken text after a successful analysis so the box
            # doesn't stay pre-filled with old speech next time.
            st.session_state.spoken_text = ""

# =================================================================
# TAB 2: MOOD HISTORY
# =================================================================
with tab2:
    st.subheader("📊 Your Mood Over Time")

    if not st.session_state.history:
        st.info("No entries yet. Analyze some text in the 'Detect Emotion' tab to start tracking your mood.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, width="stretch")

        emotion_counts = df["Emotion"].value_counts()
        fig2 = go.Figure(data=[go.Pie(
            labels=[f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in emotion_counts.index],
            values=emotion_counts.values,
            hole=0.4
        )])
        fig2.update_layout(
            title="Mood Distribution",
            paper_bgcolor="#0E1117",
            font=dict(color="white")
        )
        st.plotly_chart(fig2, width="stretch")

        # --- Mood Calendar Heatmap (date x emotion) ---
        st.markdown("### 🗓️ Mood Calendar Heatmap")
        pivot = df.groupby(["Date", "Emotion"]).size().reset_index(name="Count")
        heatmap_data = pivot.pivot(index="Emotion", columns="Date", values="Count").fillna(0)

        fig3 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=[f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in heatmap_data.index],
            colorscale=[[0, "#161A22"], [1, "#FF6EC7"]],
            showscale=True
        ))
        fig3.update_layout(
            title="How your mood shifts by day",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161A22",
            font=dict(color="white"),
            height=350
        )
        fig3.update_xaxes(type="category")
        st.plotly_chart(fig3, width="stretch")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Mood History as CSV",
            data=csv_data,
            file_name=f"mood_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

st.markdown("---")
st.caption("⚠️ This tool uses AI for informational/self-reflection purposes and is not a substitute for professional mental health support.")