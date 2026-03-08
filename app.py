# app.py

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
#  SECRETS / CONFIG  (Change this before production!)
# ────────────────────────────────────────────────

AZURE_OPENAI_API_KEY    = "d61f68e18b894a36a48063cd1fe6a457"   # ← DANGER: DO NOT COMMIT
AZURE_ENDPOINT          = "https://hkust.azure-api.net"
AZURE_API_VERSION       = "2025-02-01-preview"
AZURE_MODEL             = "gpt-4o-mini"

SEMANTIC_MODEL          = "all-MiniLM-L6-v2"

MAX_JOBS                = 10
MAX_PROFILES            = 10

RANDOM_JOBS             = 100
RANDOM_PROFILES         = 100

# ────────────────────────────────────────────────
#  Initialize clients & models (cached)
# ────────────────────────────────────────────────

st.set_page_config(
    page_title = "CareerBridge AI",
    page_icon = "🌉",
    layout = "wide"
)

@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION
    )

@st.cache_resource
def get_semantic_model():
    return SentenceTransformer(SEMANTIC_MODEL)

client = get_openai_client()
embedder = get_semantic_model()

# ────────────────────────────────────────────────
#  Helper: call Azure OpenAI chat completion
# ────────────────────────────────────────────────

def generate_text(prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
    try:
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""

# ────────────────────────────────────────────────
#  1. Parse CV (PDF or DOCX)
# ────────────────────────────────────────────────

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
            st.error("Unsupported file format. Please upload PDF or DOCX.")
            return ""
        return text.strip()
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return ""

# ────────────────────────────────────────────────
#  2. Analyze CV → summary + suggestions
# ────────────────────────────────────────────────

def analyze_cv(cv_text: str) -> tuple[str, str]:
    if not cv_text.strip():
        return "", ""

    # Summary
    prompt_summary = (
        "Summarize this CV in 4-6 professional sentences, total max 150 words. "
        "Highlight key experience, skills, and career goals.\n\n"
        "CV: " + cv_text
    )
    cv_summary = generate_text(prompt_summary, max_tokens=200)

    # Suggestions
    prompt_suggestions = (
        "Give exactly 5 concrete, actionable suggestions to improve this CV for tech/job applications. "
        "Use bullet points. Focus on keywords, achievements, structure. "
        "Each suggestion should be max 50 words.\n\n"
        "CV: " + cv_text
    )
    cv_suggestions = generate_text(prompt_suggestions, max_tokens=200)

    return cv_summary, cv_suggestions

# ────────────────────────────────────────────────
#  3. Load jobs / profiles
# ────────────────────────────────────────────────
#@st.cache_data
def load_jobs() -> pd.DataFrame:
    path = "jobs.csv"
    if not os.path.exists(path):
        st.error("jobs.csv not found. Please load sample data jobs.csv")
        return pd.DataFrame(columns=["id","company","title","location","description","link"])
    try:
        df = pd.read_csv(path)
        df['description'] = df['description'].fillna('')
        required = {"id","company","title","location","description","link"}
        if not required.issubset(df.columns):
            st.warning("jobs.csv is missing some expected columns")
        return df
    except Exception as e:
        st.error(f"Error reading jobs.csv: {e}")
        return pd.DataFrame()

#@st.cache_data
def load_profiles() -> pd.DataFrame:
    path = "profiles.json"
    if not os.path.exists(path):
        st.error("profiles.json not found. Please load sample data profiles.json")
        return pd.DataFrame(columns=["public_identifier","full_name","country","city","headline","summary"])
    try:
        df = pd.read_json(path, lines=True)
        text_cols = ['headline', 'summary', 'name', 'country', 'city', 'id']
        
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype('string').fillna('')

        df = df.rename(columns={"public_identifier": "id", "full_name": "name"})
    
        required = {"id","name","country","city","headline","summary"}
        if not required.issubset(df.columns):
            st.warning("profiles.json is missing some expected columns")
        return df
    except Exception as e:
        st.error(f"Error reading profiles.json: {e}")
        return pd.DataFrame()

# ────────────────────────────────────────────────
#  4. Semantic matching
# ────────────────────────────────────────────────

def match_jobs(cv_summary: str, job_interest: str, df_jobs: pd.DataFrame) -> pd.DataFrame:
    if df_jobs.empty or not cv_summary.strip():
        return pd.DataFrame()

    df = df_jobs.copy()

    # Create combined text for jobs
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("")
    ).str.strip()

    # Embeddings
    cv_emb        = embedder.encode(cv_summary, convert_to_tensor=True)
    interest_emb  = embedder.encode(job_interest, convert_to_tensor=True)
    query_emb     = (cv_emb + interest_emb) / 2   # simple average

    job_embs      = embedder.encode(df["combined_text"].tolist(), convert_to_tensor=True)

    cos_scores = util.cos_sim(query_emb, job_embs)[0].cpu().numpy()
    df["match_score"] = np.round(cos_scores * 100, 2)

    # Sort & take top 10
    df = df.sort_values("match_score", ascending=False).head(MAX_JOBS).reset_index(drop=True)

    # Generate "why suitable" reasoning
    df["reason"] = ""
    df["summary"] = ""
    for i, row in df.iterrows():

        prompt_job_summary = (
            f"Summarize below job in max 30 words.\n\n"
            f"Job: {row["description"][:500]}\n"
        )
        job_summary = generate_text(prompt_job_summary, temperature=0.7)
        df.at[i, "summary"] = job_summary

        prompt = (
            f"Why this job is suitable for me? Comment within 50 words.\n\n"
            f"Job title: {row['title']}\n"
            f"Job description: {job_summary}\n"
            f"My CV summary: {cv_summary}\n"
            f"My job interests: {job_interest}\n"
        )
        df.at[i, "reason"] = generate_text(prompt, max_tokens=220, temperature=0.7)

    return df

def match_profiles(cv_summary: str, job_interest: str, df_profiles: pd.DataFrame) -> pd.DataFrame:
    if df_profiles.empty or not cv_summary.strip():
        return pd.DataFrame()

    df = df_profiles.copy()

    df["combined_text"] = (
        df["headline"].fillna("") + " " +
        df["summary"].fillna("")
    ).str.strip()

    cv_emb       = embedder.encode(cv_summary, convert_to_tensor=True)
    interest_emb = embedder.encode(job_interest, convert_to_tensor=True)
    query_emb    = (cv_emb + interest_emb) / 2

    profile_embs = embedder.encode(df["combined_text"].tolist(), convert_to_tensor=True)

    cos_scores = util.cos_sim(query_emb, profile_embs)[0].cpu().numpy()
    df["match_score"] = np.round(cos_scores * 100, 2)

    df = df.sort_values("match_score", ascending=False).head(MAX_PROFILES).reset_index(drop=True)

    # Generate reason & greeting for top 10
    df["reason"]   = ""
    df["greeting"] = ""

    for i, row in df.head(MAX_PROFILES).iterrows():
        # summary
        prompt_profile_summary = (
            f"Summarize below profile in max 30 words.\n\n"
            f"Job: {row["summary"][:500]}\n"
        )
        profile_summary = generate_text(prompt_profile_summary, temperature=0.7)
        df.at[i, "summary"] = profile_summary

        # Reason
        prompt_reason = (
            f"Why this mentor can help me in my career path? comment in max 50 words\n"
            f"Mentor headline: {row['headline']}\n"
            f"Mentor summary: {profile_summary}\n"
            f"My CV summary: {cv_summary}\n"
            f"My job interests: {job_interest}"
        )
        df.at[i, "reason"] = generate_text(prompt_reason, max_tokens=100, temperature=0.7)

        # Greeting
        prompt_greeting = (
            f"Write a short, warm, professional first-message (max 5 sentences) "
            f"to invite this mentor for a 15-min virtual coffee chat.\n"
            f"Mentor's name: {row['name']}\n"
            f"Mentor's job: {row['headline']}\n"
            f"Mentor's summary: {row.get('summary','')}\n\n"
            f"My CV summary: {cv_summary}\n"
            f"My job interests: {job_interest}\n\n"
            "Tone: respectful, concise, genuine. End with a clear call-to-action."
        )
        df.at[i, "greeting"] = generate_text(prompt_greeting, max_tokens=100, temperature=0.7)

    return df

# ────────────────────────────────────────────────
#  MAIN STREAMLIT APP
# ────────────────────────────────────────────────

def main():

    # ── Header ───────────────────────────────────────
    st.title("🌉 CareerBridge AI")
    st.header("Bridge the gap to your dream career")
    st.markdown(
        "Upload your CV, tell me about your interests in searching jobs. "
        "I will find **Jobs** and **Mentors** from LinkedIn for you."
    )
    st.divider()

    # ── Inputs ───────────────────────────────────────
    col_left, col_right = st.columns([5,5])

    with col_left:
        st.subheader("Step 1: 📄 Upload your CV")
        uploaded_file = st.file_uploader(
            "PDF or DOCX only",
            type=["pdf", "docx"],
            help="Upload your resume / CV"
        )

    with col_right:
        st.subheader("Step 2: 🧭 Your job Interest")
        job_interest = st.text_area(
            label="Describe the roles, industries, technologies or locations you're interested in",
            height=140,
            placeholder="Example:\n• AI / Machine Learning Engineer\n• Remote or Hong Kong\n• Python, PyTorch, LLM experience"
        ).strip()

    if not uploaded_file or not job_interest:
        st.info("Please upload your CV and describe your job interests to start matching.")
        st.stop()

    st.divider()

    # ── Processing ───────────────────────────────────
    with st.spinner("Reading CV ..."):
        cv_text = parse_cv(uploaded_file)

    if not cv_text:
        st.stop()

    # ── CV Analysis & Suggestions ────────────────────
    with st.spinner("Analyzing your CV ..."):
        cv_summary, cv_suggestions = analyze_cv(cv_text)

    col1, col2 = st.columns([5,5])

    with col1:
        st.subheader("📊 Analysis")
        st.markdown(cv_summary or "*No summary generated*")

    with col2:
        st.subheader("💡 Suggestions to improve CV")
        st.markdown(cv_suggestions or "*No suggestions generated&")

    st.divider()

    # ── Load datasets ────────────────────────────────
    with st.spinner("Loading LinkedIn datasets ..."):
        df_jobs     = load_jobs().sample(n=RANDOM_JOBS, random_state=1)
        df_profiles = load_profiles().sample(n=RANDOM_PROFILES, random_state=1)

    if df_jobs.empty and df_profiles.empty:
        st.error("No job or profile data available. Cannot perform matching.")
        st.stop()

    # ── Matching ─────────────────────────────────────
    with st.spinner("Finding best job & mentor matches ... (this may take 30–90 seconds)"):
        df_matched_jobs     = match_jobs(cv_summary, job_interest, df_jobs)
        df_matched_profiles = match_profiles(cv_summary, job_interest, df_profiles)

    # ── Results ──────────────────────────────────────
    col_jobs, col_mentors = st.columns([5,5])

    with col_jobs:
        st.subheader("🔍 Job Matches on LinkedIn")

        if df_matched_jobs.empty:
            st.info("No job matches found.")
        else:
            for _, row in df_matched_jobs.iterrows():
                with st.expander(f"{row['title']} - Scores: {np.round(row['match_score'], 2)}%"):
                    st.markdown(f"**Company:** {row.get('company','–')}")
                    st.markdown(f"**Location:** {row.get('location','–')}")
                    st.markdown(f"**Description:**\n{row.get('summary','–')}")
                    st.markdown(f"**Why suitable?**\n{row.get('reason','–')}")
                    st.link_button("🔗 View LinkedIn Job", row['link'], use_container_width=False)

    with col_mentors:
        st.subheader("👥 Career Path Mentors on LinkedIn")

        if df_matched_profiles.empty:
            st.info("No mentor matches found.")
        else:
            for i, row in df_matched_profiles.iterrows():
                with st.expander(f"{row['name']} - Scores: {np.round(row['match_score'], 2)}%"):
                    st.markdown(f"**Location:** {row.get('city','–')}, {row.get('country','–')}")
                    st.markdown(f"**Job:**\n{row.get('headline','–')}")
                    st.markdown(f"**Summary:**\n{row.get('summary','–')}")
                    st.markdown(f"**Why suggest this mentor?**: {row.get('reason','–')}")
                    st.link_button("🔗 View LinkedIn Profile", f"https://www.linkedin.com/in/{row['id']}/", use_container_width=False)
                    st.markdown(f"**☕ Coffee Chat Invite**:\n\n{row.get('greeting','–')}")

if __name__ == "__main__":
    main()