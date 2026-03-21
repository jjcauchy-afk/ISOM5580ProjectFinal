import streamlit as st
import os
import pandas as pd
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pypdf
import docx2txt
from pathlib import Path
import time
import datetime
import pytz
from dotenv import load_dotenv

# ────────────────────────────────────────────────
#  CSS
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
load_dotenv()

def get_config(key):
    # Try to get from Streamlit secrets first
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fallback to environment variables 
    return os.getenv(key, "")

AZURE_OPENAI_API_KEY = get_config("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = get_config("AZURE_ENDPOINT")
AZURE_API_VERSION = get_config("AZURE_API_VERSION")
AZURE_MODEL = get_config("AZURE_MODEL")

SEMANTIC_MODEL = "all-MiniLM-L6-v2"
MAX_JOBS = 10
MAX_PROFILES = 10
JOBS_PER_PAGE = 5
PROFILES_PER_PAGE = 5

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
if 'j_emb' not in st.session_state:
    st.session_state.j_emb = None
if 'p_emb' not in st.session_state:
    st.session_state.p_emb = None
if 'target_job_desc' not in st.session_state:
    st.session_state.target_job_desc = ""
if 'cv_improvements' not in st.session_state:
    st.session_state.cv_improvements = ""
if 'candidate_improvements' not in st.session_state:
    st.session_state.candidate_improvements = ""
if 'jobs_page' not in st.session_state:
    st.session_state.jobs_page = 1
if 'profiles_page' not in st.session_state:
    st.session_state.profiles_page = 1

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
#  BATCH OPENAI CALLS
# ────────────────────────────────────────────────
def generate_text(prompt, max_tokens = 800, temperature = 0.5) -> str:
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

def generate_batch(prompts, max_tokens_per = 100, temperature = 0.5):
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
            temperature=temperature
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
def cv_parse(uploaded_file):
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

def cv_summary(cv_text):
    if not cv_text:
        return "", ""
    sum_txt = generate_text("Summarize CV in 4-6 sentences (max 150 words):\n"+cv_text, 800)
    return sum_txt

def cv_suggestion(cv_text):
    if not cv_text:
        return "", ""
    sug_txt = generate_text("5 CV improvements (max 50 words each):\n"+cv_text, 800)
    return sug_txt

@st.cache_data
def load_jobs_data():
    try:
        df = pd.read_csv("dataset_jobs.csv").fillna("")
        return df
    except Exception as e:
        st.error(f"Failed to load jobs data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_profiles_data():
    try:
        df = pd.read_csv("dataset_profiles.csv").fillna("")
        return df
    except Exception as e:
        st.error(f"Failed to load profiles data: {e}")
        return pd.DataFrame()

# ────────────────────────────────────────────────
#  JOB MATCHING
# ────────────────────────────────────────────────
def match_jobs_auto(cv_summary, job_interest):
    if st.session_state.df_jobs.empty:
        st.error("Job not available in session.")
        return pd.DataFrame(), 0
    df = st.session_state.df_jobs.copy()

    # Compute job embeddings 
    if st.session_state.j_emb is None:
        with st.spinner("Computing job embeddings (done only once per session)..."):
            start = time.time()
            texts = (df["position"] + " " + df["description"] + " " + df["requirement"]).tolist()
            st.session_state.j_emb = embedder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32          
            )
            j_emb_time = round(time.time() - start, 2)
            st.success(f"Job embeddings ready ({j_emb_time}s)")
    
    # Semantic search
    semantic_start = time.time()
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = (cv_emb + int_emb)/2
    j_emb = st.session_state.j_emb

    df["match_score"] = np.round(util.cos_sim(q_emb, j_emb)[0].cpu().numpy() * 100, 2)
    df = df.sort_values("match_score", ascending=False).head(MAX_JOBS).reset_index(drop=True)
    semantic_time = round(time.time() - semantic_start, 2)

    # OpenAI
    openai_start = time.time()
    prompts = []
    for _, row in df.iterrows():
        prompts.append(f"Summarize job in 50 words:\n{row['description'][:1000]}")
        prompts.append(f"Why fit? 50 words:\nJob: {row['position']}\nMy CV: {cv_summary}\nInterests: {job_interest}")
    responses = generate_batch(prompts, max_tokens_per=60)
    # Add empty columns for summary and reason
    df["job_summary"] = ""
    df["fit_reason"] = ""
    
    summaries = []
    reasons = []
    for i in range(0, len(responses), 2):
        summaries.append(responses[i] if i < len(responses) else "")
        reasons.append(responses[i+1] if (i+1) < len(responses) else "")
    df["summary"] = summaries
    df["reason"] = reasons
    openai_time = round(time.time() - openai_start, 2)
    return df, semantic_time

    return df, semantic_time, openai_time

def match_profiles_auto(cv_summary, job_interest):
    if st.session_state.df_profiles.empty:
        st.error("Profiles not available in session.")
        return pd.DataFrame(), 0
    df = st.session_state.df_profiles.copy()

    # Compute profile embeddings only once
    if st.session_state.p_emb is None:
        with st.spinner("Computing profile embeddings (done only once per session)..."):
            start = time.time()
            texts = (st.session_state.df_profiles["position"].fillna("") + " " + st.session_state.df_profiles["about"].fillna("")).tolist()
            st.session_state.p_emb = embedder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32
            )
            p_emb_time = round(time.time() - start, 2)
            st.success(f"Profile embeddings ready ({p_emb_time}s)")

    # Semantic search
    semantic_start = time.time()
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = (cv_emb + int_emb)/2
    p_emb = st.session_state.p_emb

    df["match_score"] = np.round(util.cos_sim(q_emb, p_emb)[0].cpu().numpy() * 100, 2)
    df = df.sort_values("match_score", ascending=False).head(MAX_PROFILES).reset_index(drop=True)
    semantic_time = round(time.time() - semantic_start, 2)

    # Add empty columns for summary, reason, greeting
    df["profile_summary"] = ""
    df["fit_reason"] = ""
    df["greeting"] = ""
    
    return df, semantic_time

# ────────────────────────────────────────────────
#  PAGE NAVIGATION
# ────────────────────────────────────────────────
def render_jobs_page():
    st.header("Job Matches")
    if not st.session_state.cv_summary:
        st.warning("Upload CV first")
        return

    if st.session_state.df_jobs.empty:
        st.session_state.df_jobs = load_jobs_data()
    
    if st.session_state.matched_jobs.empty:
        with st.spinner("Executing task...", show_time=True):
            matched_jobs_df, semantic_time = match_jobs_auto(
                st.session_state.cv_summary,
                st.session_state.job_interest
            )
            st.session_state.matched_jobs = matched_jobs_df
            st.success(f"Matched jobs processed - Semantic Search: {semantic_time}s")

    df = st.session_state.matched_jobs

    if df.empty:
        st.info("No job matches found. Update your CV or job interest.")
        return

    # Pagination
    total_jobs = len(df)
    total_pages = int(np.ceil(total_jobs / JOBS_PER_PAGE))
    page = st.session_state.jobs_page

    start_idx = (page - 1) * JOBS_PER_PAGE
    end_idx = min(start_idx + JOBS_PER_PAGE, total_jobs)
    df_PAGE = df.iloc[start_idx:end_idx]

    # Compute OpenAI for missing summaries/reasons
    indices_to_compute = []
    for idx in df_PAGE.index:
        if df.at[idx, 'job_summary'] == "" or df.at[idx, 'fit_reason'] == "":
            indices_to_compute.append(idx)

    if indices_to_compute:
        with st.spinner("Generating job details..."):
            openai_start = time.time()
            prompts = []
            for idx in indices_to_compute:
                row = df.loc[idx]
                prompts.append(f"Summarize job in 50 words:\n{row['description'][:1000]}")
                prompts.append(f"Why fit? 50 words:\nJob: {row['position']}\nMy CV: {st.session_state.cv_summary}\nInterests: {st.session_state.job_interest}")
            responses = generate_batch(prompts, max_tokens_per=60)
            for i, idx in enumerate(indices_to_compute):
                df.at[idx, 'job_summary'] = responses[i*2] if i*2 < len(responses) else ""
                df.at[idx, 'fit_reason'] = responses[i*2 + 1] if i*2 + 1 < len(responses) else ""
            openai_time = round(time.time() - openai_start, 2)
            st.success(f"Job details generated in {openai_time}s")

    for _, row in df_PAGE.iterrows():
        with st.expander(f"**{row['position']}** - Score: {round(row['match_score'],2)}%"):
            col1, col2 = st.columns([3, 1])  
            with col1:
                st.write(f"**Company**: {row.get('company')}")
                st.write(f"**Location**: {row.get('location')}")
                st.write(f"**Summary**:  \n{row.get('job_summary')}")
                st.write(f"**Why Fit?**  \n{row.get('fit_reason')}")
            
            with col2:
                if "url" in row and row['url']:
                    linkedin_url = row['url']
                    st.link_button(
                        label="View Job on LinkedIn",
                        url=linkedin_url,
                        use_container_width=True,
                        type="primary"
                    )
                if st.button("Tailor CV for This Job", key=f"tailor_{_}"):
                    st.session_state.target_job_desc = "Job Title: " + row['position'] + "\n\n" + row['description']
                    st.session_state.current_page = "target_job"
                    st.rerun()

    # Page navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.jobs_page > 1:
            if st.button("Previous", key="prev_jobs"):
                st.session_state.jobs_page -= 1
                st.rerun()

    with col2:
        if st.session_state.jobs_page < total_pages:
            if st.button("Next", key="next_jobs"):
                st.session_state.jobs_page += 1
                st.rerun()

def render_profiles_page():
    st.header("Profile Matches")
    if not st.session_state.cv_summary:
        st.warning("Upload CV first")
        return

    if st.session_state.df_profiles.empty:
        df = load_profiles_data()
        df = df.dropna(subset=['position','about']).query("position!='' & about!=''")
        st.session_state.df_profiles = df
    
    if st.session_state.matched_profiles.empty:
        with st.spinner("Executing task...", show_time=True):
            matched_profiles_df, semantic_time = match_profiles_auto(
                st.session_state.cv_summary,
                st.session_state.job_interest
            )
            st.session_state.matched_profiles = matched_profiles_df
            st.success(f"Matched mentors processed - Semantic Search: {semantic_time}s")

    df = st.session_state.matched_profiles

    if df.empty:
        st.info("No profile matches found. Update your CV or job interest.")
        return

    # Pagination
    total_profiles = len(df)
    total_pages = int(np.ceil(total_profiles / PROFILES_PER_PAGE))
    page = st.session_state.profiles_page

    start_idx = (page - 1) * PROFILES_PER_PAGE
    end_idx = min(start_idx + PROFILES_PER_PAGE, total_profiles)
    df_PAGE = df.iloc[start_idx:end_idx]

    # Compute OpenAI for missing summaries/reasons/greet
    indices_to_compute = []
    for idx in df_PAGE.index:
        if df.at[idx, 'profile_summary'] == "" or df.at[idx, 'fit_reason'] == "" or df.at[idx, 'greeting'] == "":
            indices_to_compute.append(idx)

    if indices_to_compute:
        with st.spinner("Generating profile details..."):
            openai_start = time.time()
            prompts = []
            for idx in indices_to_compute:
                row = df.loc[idx]
                prompts.append(f"Summarize profile in 50 words:\n{row['about'][:1000]}")
                prompts.append(f"Why fit? 50 words:\nJob: {row['position']}\nMy CV: {st.session_state.cv_summary}\nInterests: {st.session_state.job_interest}")
                prompts.append(f"Generate a greeting message for networking:\n{row['about'][:500]}")
            responses = generate_batch(prompts, max_tokens_per=60)
            for i, idx in enumerate(indices_to_compute):
                df.at[idx, 'profile_summary'] = responses[i*3] if i*3 < len(responses) else ""
                df.at[idx, 'fit_reason'] = responses[i*3 + 1] if i*3 + 1 < len(responses) else ""
                df.at[idx, 'greeting'] = responses[i*3 + 2] if i*3 + 2 < len(responses) else ""
            openai_time = round(time.time() - openai_start, 2)
            st.success(f"Profile details generated in {openai_time}s")

    for _, row in df_PAGE.iterrows():
        with st.expander(f"**{row['position']}** - Score: {round(row['match_score'],2)}%"):
            col1, col2 = st.columns([3, 1])  
            with col1:
                st.write(f"**Name**: {row.get('name')}")
                st.write(f"**Company**: {row.get('company')}")
                st.write(f"**Location**: {row.get('location')}")
                st.write(f"**Summary**:  \n{row.get('profile_summary')}")
                st.write(f"**Why Fit?**  \n{row.get('fit_reason')}")
                st.write(f"**Connecting Message**:  \n{row.get('greeting')}")
            
            with col2:
                linkedin_url = f"https://www.linkedin.com/in/{row['linkedin_id']}"
                st.link_button(
                    label="View Profile on LinkedIn",
                    url=linkedin_url,
                    use_container_width=True,
                    type="primary"
                )
                if st.button("Engage with Profile", key=f"engage_{_}"):
                    st.session_state.target_job_desc = "Job Title: " + row['position'] + "\n\n" + row['about']
                    st.session_state.current_page = "target_profile"
                    st.rerun()

    # Page navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.profiles_page > 1:
            if st.button("Previous", key="prev_profiles"):
                st.session_state.profiles_page -= 1
                st.rerun()

    with col2:
        if st.session_state.profiles_page < total_pages:
            if st.button("Next", key="next_profiles"):
                st.session_state.profiles_page += 1
                st.rerun()

# Map page names to functions
pages = {
    "upload_cv": page_upload_cv,
    "cv_suggestions": page_cv_suggestions,
    "matched_jobs": render_jobs_page,
    "matched_profiles": render_profiles_page
}

# Show the current page
if st.session_state.current_page in pages:
    pages[st.session_state.current_page]()
else:
    st.error("Page not found")
