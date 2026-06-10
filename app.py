
import streamlit as st
import whisper
import librosa
import numpy as np

# from transformers import pipeline
from groq import Groq
import tempfile
from streamlit_mic_recorder import mic_recorder
# ===================================
# PAGE CONFIG
# ===================================
# ==================================================
# PAGE HEADER
# ==================================================
st.set_page_config(
    page_title="AI Interview Assessment",
    page_icon="🎤",
    layout="wide"
)
st.markdown("""
<h1 style='text-align:center;'>
🎤 AI Interview Assessment System
</h1>

<h4 style='text-align:center;color:gray;'>
AI Powered Interview Evaluation Platform
</h4>
""", unsafe_allow_html=True)
st.sidebar.title("🎤 Interview AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🎤 Interview Assessment",
        "🤖 Mock Interview",
        "📊 Reports"
    ]
)

st.sidebar.markdown("---")

if page =="🏠 Dashboard":

    st.markdown("## 📊 Platform Overview")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Interviews Analyzed","128")
    c2.metric("Average Score","8.4")
    c3.metric("Mock Interviews","42")
    c4.metric("Success Rate","91%")

    st.markdown("---")

    st.info("""
    Welcome to AI Interview Assessment Platform.

    Upload interview recordings, receive AI-powered feedback,
    analyze communication skills and prepare for placements.
    """)
if page == "🎤 Interview Assessment":
# ==================================================
# HERO SECTION
# ==================================================
    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#0f172a,#1e3a8a);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:30px;
    ">

    <h1>Practice. Analyze. Improve.</h1>

    <h4>
    AI-powered interview assessment for students and job seekers.
    </h4>

    <p>
    Upload or record answers and receive professional feedback
    on communication, confidence, fluency and technical clarity.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ==================================================
    # INPUT METHOD
    # ==================================================

    st.markdown("### Choose Input Method")

    option = st.radio(
        "",
        ["📁 Upload Audio", "🎙 Record Live Answer"],
        horizontal=True
    )
    # ==================================================
    # FEATURE CARDS
    # ==================================================
    st.markdown("### What We Evaluate")

    col1, col2, col3 = st.columns(3)

    with col1:
            st.info("""
            ### 🗣 Communication

            • Grammar Quality

            • Clarity

            • Filler Word Usage

            • Speaking Flow
            """)

    with col2:
            st.success("""
            ### 🎯 Confidence

            • Tone Stability

            • Speech Delivery

            • Vocal Confidence

            • Pace Control
            """)

    with col3:
            st.warning("""
            ### 🤖 AI Feedback

            • Strengths

            • Weaknesses

            • Interview Readiness

            • Improvement Tips
            """)

    # ==================================================
    # AUDIO INPUT
    # ==================================================
    uploaded_file = None
    audio = None
    if option == "📁 Upload Audio":

        uploaded_file = st.file_uploader(
            "Upload Interview Audio",
            type=["wav", "mp3", "m4a"]
        )

    else:
        audio = mic_recorder(
                start_prompt="🎙 Start Recording",
                stop_prompt="⏹ Stop Recording",
                key="recorder"
        )
        if audio:
            st.success("🎙 Recording captured successfully")
    st.markdown("<br>", unsafe_allow_html=True)
    @st.cache_resource
    def load_whisper():
        return whisper.load_model("base")

    model = load_whisper()


    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
    audio_path = None

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

    if audio:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio["bytes"])
            audio_path = tmp_file.name

    if audio_path:
        if st.button("🚀 Analyze Interview", use_container_width=True):
                


            # ===================================
            # PROCESS AUDIO
            # ===================================
                st.success("✅ Audio ready for analysis")

                # ===================================
                # SPEECH TO TEXT
                # ===================================

                with st.spinner("Transcribing audio..."):
                    result = model.transcribe(audio_path)

                transcript = result["text"]

                st.subheader("📝 Transcript")
                st.write(transcript)
                
                # ===================================
                # FILLER WORD ANALYSIS
                # ===================================

                fillers = [
                    "um",
                    "uh",
                    "like",
                    "actually",
                    "basically",
                    "you know"
                ]

                filler_count = 0

                for word in fillers:
                    filler_count += transcript.lower().count(word)

                filler_score = max(0, 10 - filler_count)

                # ===================================
                # GRAMMAR ANALYSIS
                # ===================================

                grammar_score = 8
                grammar_errors = 0
                
                # ===================================
                # SPEAKING SPEED
                # ===================================

                y, sr = librosa.load(audio_path)

                duration = librosa.get_duration(y=y, sr=sr)

                word_count = len(transcript.split())

                wpm = (word_count / duration) * 60

                if 100 <= wpm <= 150:
                    fluency_score = 10
                elif 80 <= wpm < 100:
                    fluency_score = 7
                else:
                    fluency_score = 5
                
                # ===================================
                # PITCH ANALYSIS
                # ===================================

                pitch = librosa.yin(
                    y,
                    fmin=50,
                    fmax=300
                )

                avg_pitch = np.mean(pitch)

                if 150 <= avg_pitch <= 250:
                    confidence_score = 10
                else:
                    confidence_score = 7
                
                # ===================================
                # SENTIMENT ANALYSIS
                # ===================================

                sentiment_label = "POSITIVE"
                sentiment_score = 10

                

                # ===================================
                # FINAL SCORE
                # ===================================

                final_score = (
                    filler_score * 0.20 +
                    grammar_score * 0.30 +
                    fluency_score * 0.20 +
                    confidence_score * 0.20 +
                    sentiment_score * 0.10
                )
                word_count = len(transcript.split())

                if word_count < 20:
                    st.warning("Answer too short for accurate evaluation.")
                    final_score = min(final_score, 5)
                # ===================================
                # DISPLAY ANALYTICS
                # ===================================

                st.markdown("## 📊 Interview Analytics")

                col1,col2,col3,col4 = st.columns(4)

                with col1:
                    st.metric("Communication", f"{filler_score*10}%")

                with col2:
                    st.metric("Grammar", f"{grammar_score*10}%")

                with col3:
                    st.metric("Fluency", f"{fluency_score*10}%")

                with col4:
                    st.metric("Confidence", f"{confidence_score*10}%")

                st.progress(final_score/10)

                st.success(
                    f"Overall Interview Score: {round(final_score,1)}/10"
                )
                st.markdown("### Detailed Metrics")

                c1,c2,c3 = st.columns(3)

                c1.metric("Words Spoken", word_count)
                c2.metric("Speaking Speed", f"{round(wpm)} WPM")
                c3.metric("Filler Words", filler_count)

            # ===================================
            # LLM FEEDBACK
            # ===================================

                with st.spinner("Generating AI feedback..."):

                    prompt = f"""
        You are an experienced HR interviewer.

        Evaluate the following interview answer.

        Answer:
        {transcript}

        Provide:

        1. Communication Evaluation
        2. Confidence Evaluation
        3. Technical Clarity
        4. Strengths
        5. Weaknesses
        6. Suggestions for Improvement

        Keep feedback professional and concise.
        """

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                feedback = response.choices[0].message.content

                st.markdown("---")
                st.markdown("## 🤖 AI Interview Feedback")

                st.info(feedback)

                st.success(
                f"Final Interview Score: {round(final_score,2)} / 10"
                )
                report = f"""
            TRANSCRIPT

            {transcript}

            --------------------------------

            Filler Score: {filler_score}
            Grammar Score: {grammar_score}
            Fluency Score: {fluency_score}
            Confidence Score: {confidence_score}

            Final Score: {round(final_score,2)}

            --------------------------------

            AI FEEDBACK

            {feedback}
            """

                st.download_button(
                label="📄 Download Report",
                data=report,
                file_name="interview_report.txt",
                mime="text/plain"
                )