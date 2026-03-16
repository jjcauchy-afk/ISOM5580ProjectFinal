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
#  CSS (UPDATED FOR BETTER BUTTON STYLING)
# ────────────────────────────────────────────────
def add_highlighted_button_css():
    st.markdown("""
    <style>
    /* Sidebar buttons */
    div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
    }
    div[data-testid="stButton"] button:hover {
    }

    /* Uploader button */
    div[data-testid="stFileUploader"] button {
    }
    div[data-testid="stFileUploader"] button:hover {
    }

    /* Process CV button */
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
    }
    div[data-testid="stBaseButton-primary"] button:hover {
    }

    /* LinkedIn buttons */
    div[data-testid="stLinkButton"] a {
        display: inline-block;
        background-color: #0077B5;    
        color: white;                  
        padding: 10px 20px;           
        border-radius: 5px;   
        border-color: #005582; 
        transition: background-color 0.3s ease; /* Smooth transition */
    }
    div[data-testid="stLinkButton"] a p {
        font-weight: bold;             
        font-size: 16px;
        text-align:center;
    }
    div[data-testid="stLinkButton"] a:hover {
        background-color: #005582; 
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
MAX_JOBS = 3
MAX_PROFILES = 3
RANDOM_JOBS = 10
RANDOM_PROFILES = 10

# ────────────────────────────────────────────────
#  SESSION STATE
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
#  AUTO-RESET
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
#  BATCH OPENAI CALLS (SPEEDUP)
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

def generate_batch(prompts, max_tokens_per=100):
    if not client or not prompts:
        return [""] * len(prompts)
    try:
        full_prompt = (
            "Answer each question below in order. Use '|||' as separator between answers. No extra text.\n\n"
            + "\n---\n".join(prompts)
        )
        res = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens_per * len(prompts),
            temperature=0.3
        )
        parts = res.choices[0].message.content.strip().split("|||")
        parts = [p.strip() for p in parts]
        while len(parts) < len(prompts):
            parts.append("")
        return parts[:len(prompts)]
    except:
        return [""] * len(prompts)

# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────
def parse_cv(uploaded_file):
    if not uploaded_file:
        return ""
    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext == ".pdf":
            return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(uploaded_file).pages)
        elif ext in [".docx", ".doc"]:
            return docx2txt.process(uploaded_file)
    except:
        st.error("CV read error")
    return ""

def analyze_cv(cv_text):
    if not cv_text:
        return "", ""
    sum_txt = generate_text("Summarize CV in 4-6 sentences (max 150 words):\n"+cv_text, 200)
    sug_txt = generate_text("5 bullet CV improvements (max 50 words each):\n"+cv_text, 200)
    return sum_txt, sug_txt

@st.cache_data
def load_jobs_data():
    if not os.path.exists("jobs.csv"):
        return pd.DataFrame()
    df = pd.read_csv("jobs.csv").fillna("")
    return df

@st.cache_data
def load_profiles_data():
    if not os.path.exists("profiles.json"):
        return pd.DataFrame()
    df = pd.read_json("profiles.json", lines=True)
    if "public_identifier" in df.columns:
        df["id"] = df["public_identifier"]
    if "full_name" in df.columns:
        df["name"] = df["full_name"]
    return df.fillna("")

# ────────────────────────────────────────────────
#  JOB MATCHING
# ────────────────────────────────────────────────
def match_jobs_auto(cv_summary, job_interest, df_jobs):
    if df_jobs.empty or not cv_summary:
        return pd.DataFrame()
    df = df_jobs.copy()
    df["combined_text"] = df["title"] + " " + df["description"]
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = (cv_emb + int_emb)/2
    j_emb = embedder.encode(df["combined_text"].tolist(), convert_to_tensor=True)
    df["match_score"] = np.round(util.cos_sim(q_emb, j_emb)[0].cpu().numpy()*100,2)
    df = df.sort_values("match_score", ascending=False).head(MAX_JOBS).reset_index(drop=True)

    prompts = []
    for _, row in df.iterrows():
        prompts.append(f"Summarize job in 30 words:\n{row['description'][:500]}")
        prompts.append(f"Why fit? 50 words:\nJob: {row['title']}\nMy CV: {cv_summary}\nInterests: {job_interest}")
    responses = generate_batch(prompts, max_tokens_per=60)

    summaries = []
    reasons = []
    for i in range(0, len(responses), 2):
        summaries.append(responses[i] if i < len(responses) else "")
        reasons.append(responses[i+1] if (i+1) < len(responses) else "")
    df["summary"] = summaries
    df["reason"] = reasons
    return df

# ────────────────────────────────────────────────
#  MENTOR MATCHING
# ────────────────────────────────────────────────
def match_profiles_auto(cv_summary, job_interest, df_profiles):
    if df_profiles.empty or not cv_summary:
        return pd.DataFrame()
    df = df_profiles.copy()
    df["combined_text"] = df["headline"].fillna("") + " " + df["summary"].fillna("")
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = (cv_emb + int_emb)/2
    p_emb = embedder.encode(df["combined_text"].tolist(), convert_to_tensor=True)
    df["match_score"] = np.round(util.cos_sim(q_emb, p_emb)[0].cpu().numpy()*100,2)
    df = df.sort_values("match_score", ascending=False).head(MAX_PROFILES).reset_index(drop=True)

    prompts = []
    for _, row in df.iterrows():
        prompts.append(f"Summarize mentor profile in 30 words:\n{row['summary'][:500]}")
        prompts.append(f"Why mentor match? 50 words:\n{row['headline']}\nMy CV: {cv_summary}")
        prompts.append(f"30-word LinkedIn message to {row['name']} for 15min career chat.")
    res = generate_batch(prompts, max_tokens_per=60)

    summaries = []
    reasons = []
    greetings = []
    for i in range(0, len(res), 3):
        summaries.append(res[i] if i < len(res) else "")
        reasons.append(res[i+1] if (i+1) < len(res) else "")
        greetings.append(res[i+2] if (i+2) < len(res) else "")
    df["summary"] = summaries
    df["reason"] = reasons
    df["greeting"] = greetings
    return df

# ────────────────────────────────────────────────
#  PAGES — LINK BUTTONS (IMPROVED UI)
# ────────────────────────────────────────────────
def page_upload_cv():
    st.title("🌉 CareerBridge AI")
    st.header("Upload CV and Job Interests")
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader("PDF/DOCX only", type=["pdf","docx"])
    with col2:
        st.session_state.job_interest = st.text_area("Target roles, locations, skills", placeholder="e.g., Software Engineer\nSkills: Python, Machine Learning", value=st.session_state.job_interest, height=140).strip()
    if st.button("✅ Process CV", type="primary", use_container_width=True):
        if file:
            txt = parse_cv(file)
            if txt:
                st.session_state.cv_text = txt
                st.session_state.cv_summary = analyze_cv(txt)[0]
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
        st.session_state.cv_suggestions = analyze_cv(st.session_state.cv_text)[1]
    st.markdown(st.session_state.cv_suggestions)

def page_matched_jobs():
    st.title("🔍 Matched Jobs")
    if not st.session_state.cv_summary:
        st.warning("Upload CV first")
        return
    if st.session_state.df_jobs.empty:
        st.session_state.df_jobs = load_jobs_data()
    
    if st.session_state.matched_jobs.empty:
        st.session_state.matched_jobs = match_jobs_auto(
            st.session_state.cv_summary,
            st.session_state.job_interest,
            st.session_state.df_jobs
        )
    
    for _, row in st.session_state.matched_jobs.iterrows():
        with st.expander(f"{row['title']} | {row['match_score']}%"):
            col1, col2 = st.columns([3, 1])  # Split for content + button
            with col1:
                st.write(f"**Company**: {row.get('company')}")
                st.write(f"**Location**: {row.get('location')}")
                st.write(f"**Summary**: {row.get('summary')}")
                st.write(f"**Fit**: {row.get('reason')}")
            
            with col2:
                # Find job URL and create button
                job_url = None
                if "job_url" in row and row["job_url"]:
                    job_url = row["job_url"]
                elif "url" in row and row["url"]:
                    job_url = row["url"]
                elif "link" in row and row["link"]:
                    job_url = row["link"]
                elif "job_link" in row and row["job_link"]:
                    job_url = row["job_link"]

                if job_url:
                    # LinkedIn-style button for job link
                    st.link_button(
                        label="View Job on LinkedIn",
                        url=job_url,
                        use_container_width=True,
                        type="primary"
                    )

def page_matched_profiles():
    st.title("👥 Career Mentors")
    if not st.session_state.cv_summary:
        st.warning("Upload CV first")
        return
    if st.session_state.df_profiles.empty:
        df = load_profiles_data()
        df = df.dropna(subset=['headline','summary']).query("headline!='' & summary!=''")
        st.session_state.df_profiles = df
    
    if st.session_state.matched_profiles.empty:
        st.session_state.matched_profiles = match_profiles_auto(
            st.session_state.cv_summary,
            st.session_state.job_interest,
            st.session_state.df_profiles
        )
    
    for _, row in st.session_state.matched_profiles.iterrows():
        with st.expander(f"{row['name']} | {row['match_score']}%"):
            col1, col2 = st.columns([3, 1])  # Split for content + button
            with col1:
                st.write(f"**Headline**: {row.get('headline')}")
                st.write(f"**Profile Summary**: {row.get('summary')}")
                st.write(f"**Fit**: {row.get('reason')}")
                st.divider()
                st.markdown(f"**☕ Message**: {row.get('greeting')}")
            
            with col2:
                # Create LinkedIn profile button
                if "public_identifier" in row and row["public_identifier"]:
                    linkedin_url = f"https://linkedin.com/in/{row['public_identifier']}"
                    st.link_button(
                        label="View LinkedIn Profile",
                        url=linkedin_url,
                        use_container_width=True,
                        type="primary"
                    )

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

    pages = {
        "upload_cv": page_upload_cv,
        "cv_suggestions": page_cv_suggestions,
        "matched_jobs": page_matched_jobs,
        "matched_profiles": page_matched_profiles
    }
    pages[st.session_state.current_page]()

if __name__ == "__main__":
    main()