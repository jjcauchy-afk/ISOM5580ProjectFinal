import streamlit as st
import os
import pandas as pd
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pypdf
import docx2txt
from pathlib import Path

# ────────────────────────────────────────────────
#  CSS (LEAVE BUTTON COLOR AS-IS - NO CHANGE)
# ────────────────────────────────────────────────
def add_highlighted_button_css():
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] button {
        background-color: #f02d2d !important;
        color: white !important;
        margin-bottom: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", "")
AZURE_API_VERSION = st.secrets.get("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_MODEL = st.secrets.get("AZURE_MODEL", "")

SEMANTIC_MODEL = "all-MiniLM-L6-v2"
MAX_JOBS = 10
MAX_PROFILES = 5
RANDOM_JOBS = 100
RANDOM_PROFILES = 100

# ────────────────────────────────────────────────
#  SESSION STATE (KEEPS CV WHEN SWITCHING PAGES)
# ────────────────────────────────────────────────
st.set_page_config(page_title="CareerBridge AI", page_icon="🌉", layout="wide")

if 'cv_text' not in st.session_state:
    st.session_state.cv_text = ""
if 'cv_summary' not in st.session_state:
    st.session_state.cv_summary = ""
if 'cv_suggestions' not in st.session_state:
    st.session_state.cv_suggestions = ""
if 'job_interest' not in st.session_state:
    st.session_state.job_interest = ""
if 'matched_jobs' not in st.session_state:
    st.session_state.matched_jobs = pd.DataFrame()
if 'matched_profiles' not in st.session_state:
    st.session_state.matched_profiles = pd.DataFrame()
if 'df_jobs' not in st.session_state:
    st.session_state.df_jobs = pd.DataFrame()
if 'df_profiles' not in st.session_state:
    st.session_state.df_profiles = pd.DataFrame()
if 'current_page' not in st.session_state:
    st.session_state.current_page = "upload_cv"

# ────────────────────────────────────────────────
#  AUTO-RESET WHEN NEW CV OR NEW JOB INTEREST
# ────────────────────────────────────────────────
if "last_cv" not in st.session_state:
    st.session_state.last_cv = ""
if "last_interest" not in st.session_state:
    st.session_state.last_interest = ""

current_cv = st.session_state.cv_text
current_interest = st.session_state.job_interest

if current_cv != st.session_state.last_cv or current_interest != st.session_state.last_interest:
    st.session_state.cv_suggestions = ""
    st.session_state.matched_jobs = pd.DataFrame()
    st.session_state.matched_profiles = pd.DataFrame()
    st.session_state.df_jobs = pd.DataFrame()
    st.session_state.df_profiles = pd.DataFrame()
    st.session_state.last_cv = current_cv
    st.session_state.last_interest = current_interest

# ────────────────────────────────────────────────
#  MODELS
# ────────────────────────────────────────────────
@st.cache_resource
def get_openai_client():
    if not AZURE_OPENAI_API_KEY or not AZURE_ENDPOINT:
        st.error("Azure OpenAI credentials missing!")
        return None
    return AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VERSION)

@st.cache_resource
def get_semantic_model():
    return SentenceTransformer(SEMANTIC_MODEL)

client = get_openai_client()
embedder = get_semantic_model()

# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────
def generate_text(prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
    if not client:
        return ""
    try:
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return ""

def parse_cv(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_ext = Path(uploaded_file.name).suffix.lower()
    try:
        if file_ext == ".pdf":
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif file_ext in [".docx", ".doc"]:
            text = docx2txt.process(uploaded_file)
        else:
            st.error("Only PDF/DOCX")
            return ""
        return text.strip()
    except Exception as e:
        st.error(f"File read error: {e}")
        return ""

def analyze_cv(cv_text: str) -> tuple[str, str]:
    if not cv_text.strip():
        return "", ""
    prompt_sum = "Summarize this CV in 4-6 sentences, max 150 words.\n\nCV: " + cv_text
    sum_txt = generate_text(prompt_sum, max_tokens=200)

    prompt_sug = "Give exactly 5 actionable CV improvement points (bullet). Max 50 words each.\n\nCV: " + cv_text
    sug_txt = generate_text(prompt_sug, max_tokens=200)
    return sum_txt, sug_txt

@st.cache_data
def load_jobs_data():
    path = "jobs.csv"
    if not os.path.exists(path):
        st.warning("jobs.csv not found!")
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception as e:
        st.error(f"Jobs load error: {e}")
        return pd.DataFrame()

@st.cache_data
def load_profiles_data():
    path = "profiles.json"
    if not os.path.exists(path):
        st.warning("profiles.json not found!")
        return pd.DataFrame()
    try:
        df = pd.read_json(path, lines=True)
        df = df.rename(columns={"public_identifier": "id", "full_name": "name"})
        return df.fillna("")
    except Exception as e:
        st.error(f"Profiles load error: {e}")
        return pd.DataFrame()

def match_jobs_auto(cv_summary, job_interest, df_jobs):
    if df_jobs.empty or not cv_summary or not embedder:
        return pd.DataFrame()
    df = df_jobs.copy()
    df["combined_text"] = df["title"] + " " + df["description"]
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = (cv_emb + int_emb)/2
    j_emb = embedder.encode(df["combined_text"].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(q_emb, j_emb)[0].cpu().numpy()
    df["match_score"] = np.round(scores*100,2)
    df = df.sort_values("match_score", ascending=False).head(MAX_JOBS).reset_index(drop=True)

    df["summary"] = ""
    df["reason"] = ""
    for i, row in df.iterrows():
        s = generate_text(f"Summarize job in 30 words:\n{row['description'][:500]}", temperature=0.6)
        df.at[i,"summary"] = s
        r = generate_text(f"Why fit? 50 words:\nJob: {row['title']}\n{s}\nMy CV: {cv_summary}\nMy interests: {job_interest}", max_tokens=220)
        df.at[i,"reason"] = r
    return df

def match_profiles_auto(cv_summary, job_interest, df_profiles):
    if df_profiles.empty or not cv_summary or not embedder:
        return pd.DataFrame()
    df = df_profiles.copy()
    df["combined_text"] = df["headline"] + " " + df["summary"]
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = (cv_emb + int_emb)/2
    p_emb = embedder.encode(df["combined_text"].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(q_emb, p_emb)[0].cpu().numpy()
    df["match_score"] = np.round(scores*100,2)
    df = df.sort_values("match_score", ascending=False).head(MAX_PROFILES).reset_index(drop=True)

    df["summary"] = ""
    df["reason"] = ""
    df["greeting"] = ""
    for i, row in df.iterrows():
        s = generate_text(f"Summarize profile in 30 words:\n{row['summary'][:500]}", temperature=0.6)
        df.at[i,"summary"] = s
        r = generate_text(f"Why this mentor? 50 words:\n{row['headline']}\n{s}\nMy CV: {cv_summary}\nMy interests: {job_interest}", max_tokens=100)
        df.at[i,"reason"] = r
        g = generate_text(
            f"Write a short, warm, professional first-message (max 30 words) to contact the mentor in LinkedIn to invite for a 15-min coffee chat for career advice.\n\n"
            f"Mentor: {row['name']}, {row['headline']}\nMy CV: {cv_summary}\nMy interests: {job_interest}",
            max_tokens=100
        )
        df.at[i,"greeting"] = g
    return df

# ────────────────────────────────────────────────
#  PAGES
# ────────────────────────────────────────────────
def page_upload_cv():
    st.title("🌉 CareerBridge AI")
    st.header("Upload CV and Job Interests")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload CV")
        file = st.file_uploader("PDF/DOCX only", type=["pdf","docx"])

    with col2:
        st.subheader("🧭 Job Interests")
        interest = st.text_area("Target roles, locations, skills", value=st.session_state.job_interest, height=140)
        st.session_state.job_interest = interest.strip()

    if st.button("✅ Process CV", type="primary", use_container_width=True):
        if not file:
            st.warning("Upload CV first!")
            return
        with st.spinner("Reading CV..."):
            txt = parse_cv(file)
            if not txt:
                st.error("Failed to read CV")
                return
            st.session_state.cv_text = txt
        with st.spinner("Analyzing..."):
            summary, _ = analyze_cv(txt)
            st.session_state.cv_summary = summary
        st.success("✅ CV processed!")

    if st.session_state.cv_summary:
        st.divider()
        st.subheader("📊 CV Summary")
        st.markdown(st.session_state.cv_summary)

def page_cv_suggestions():
    st.title("💡 CV Suggestions")
    if not st.session_state.cv_text:
        st.warning("Upload CV first")
        return
    if not st.session_state.cv_suggestions:
        with st.spinner("Generating..."):
            _, sug = analyze_cv(st.session_state.cv_text)
            st.session_state.cv_suggestions = sug
    st.markdown(st.session_state.cv_suggestions)

def page_matched_jobs():
    st.title("🔍 Matched Jobs")
    if not st.session_state.cv_summary:
        st.warning("Upload CV first")
        return
    if st.session_state.df_jobs.empty:
        with st.spinner("Loading jobs..."):
            jobs_df = load_jobs_data()
            if jobs_df.empty:
                st.error("No jobs data")
                return
            st.session_state.df_jobs = jobs_df.sample(n=min(RANDOM_JOBS, len(jobs_df)), random_state=1)
    if st.session_state.matched_jobs.empty:
        with st.spinner("Matching jobs..."):
            st.session_state.matched_jobs = match_jobs_auto(
                st.session_state.cv_summary,
                st.session_state.job_interest,
                st.session_state.df_jobs
            )
    for _, row in st.session_state.matched_jobs.iterrows():
        with st.expander(f"{row['title']} | {row['match_score']}%"):
            st.write(f"**Company**: {row.get('company')}")
            st.write(f"**Summary**: {row.get('summary')}")
            st.write(f"**Fit**: {row.get('reason')}")

def page_matched_profiles():
    st.title("👥 Career Mentors")
    if not st.session_state.cv_summary:
        st.warning("Upload CV first")
        return
    if st.session_state.df_profiles.empty:
        with st.spinner("Loading mentors..."):
            prof_df = load_profiles_data()
            if prof_df.empty:
                st.error("No profiles")
                return
            clean = prof_df.dropna(subset=['headline','summary']).query("headline!='' & summary!=''")
            st.session_state.df_profiles = clean.sample(n=min(RANDOM_PROFILES, len(clean)), random_state=1)
    if st.session_state.matched_profiles.empty:
        with st.spinner("Matching mentors..."):
            st.session_state.matched_profiles = match_profiles_auto(
                st.session_state.cv_summary,
                st.session_state.job_interest,
                st.session_state.df_profiles
            )
    for _, row in st.session_state.matched_profiles.iterrows():
        with st.expander(f"{row['name']} | {row['match_score']}%"):
            st.write(f"**Headline**: {row.get('headline')}")
            st.write(f"**Fit**: {row.get('reason')}")
            st.divider()
            st.markdown(f"**☕ Message**: {row.get('greeting')}")

# ────────────────────────────────────────────────
#  NAVIGATION
# ────────────────────────────────────────────────
def main():
    add_highlighted_button_css()
    with st.sidebar:
        st.title("🌉 Menu")
        if st.button("📄 Upload CV", use_container_width=True):
            st.session_state.current_page = "upload_cv"
        if st.button("💡 CV Tips", use_container_width=True):
            st.session_state.current_page = "cv_suggestions"
        if st.button("🔍 Jobs", use_container_width=True):
            st.session_state.current_page = "matched_jobs"
        if st.button("👥 Mentors", use_container_width=True):
            st.session_state.current_page = "matched_profiles"

    if st.session_state.current_page == "upload_cv":
        page_upload_cv()
    elif st.session_state.current_page == "cv_suggestions":
        page_cv_suggestions()
    elif st.session_state.current_page == "matched_jobs":
        page_matched_jobs()
    elif st.session_state.current_page == "matched_profiles":
        page_matched_profiles()

if __name__ == "__main__":
    main()