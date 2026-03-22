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
import pickle

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
RANDOM_JOBS = 100
RANDOM_PROFILES = 100

# Cache paths for embeddings
J_EMB_PATH = "j_emb.pkl"
P_EMB_PATH = "p_emb.pkl"

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
def generate_text(prompt, max_tokens = 800, temperature = 0.4) -> str:
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

def generate_batch(prompts, max_tokens_per = 100, temperature = 0.4):
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
    sum_txt = generate_text("Summarize CV in max 300 words:\n"+cv_text, 800, 0.7)
    return sum_txt

def cv_suggestion(cv_text):
    if not cv_text:
        return "", ""
    sug_txt = generate_text("Suggest 3-5 CV improvements (max 50 words each):\n"+cv_text, 800, 0.7)
    return sug_txt

@st.cache_data
def load_jobs_data():
    try:
        df = pd.read_csv("dataset_jobs.csv").fillna("")
        df = df.drop_duplicates(subset=['position', 'company'])
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
        return pd.DataFrame(), 0, 0
    df = st.session_state.df_jobs.copy()

    # Compute job embeddings 
    if st.session_state.j_emb is None:
        if os.path.exists(J_EMB_PATH):
            with open(J_EMB_PATH, 'rb') as f:
                st.session_state.j_emb = pickle.load(f)
            st.success("Job embeddings loaded from cache.")
        else:
            with st.spinner("Computing job embeddings (done only once after embeddings reset, about 3 minutes)..."):
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
                # Save to cache
                with open(J_EMB_PATH, 'wb') as f:
                    pickle.dump(st.session_state.j_emb, f)
    
    # Semantic search
    semantic_start = time.time()
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = cv_emb * 0.7 + int_emb * 0.3
    j_emb = st.session_state.j_emb

    df["match_score"] = np.round(util.cos_sim(q_emb, j_emb)[0].cpu().numpy() * 100, 2)
    df = df.sort_values("match_score", ascending=False, kind='mergesort').head(MAX_JOBS).reset_index(drop=True)
    semantic_time = round(time.time() - semantic_start, 2)

    # OpenAI
    openai_start = time.time()
    prompts = []
    for _, row in df.iterrows():
        prompts.append(f"Summarize job in 50 words:\n{row['description'][:1000]}")
        prompts.append(f"Why fit? 50 words:\nJob: {row['position']}\nMy CV: {cv_summary}\nInterests: {job_interest}")
    responses = generate_batch(prompts, max_tokens_per=200)

    summaries = []
    reasons = []
    for i in range(0, len(responses), 2):
        summaries.append(responses[i] if i < len(responses) else "")
        reasons.append(responses[i+1] if (i+1) < len(responses) else "")
    df["summary"] = summaries
    df["reason"] = reasons
    openai_time = round(time.time() - openai_start, 2)
    
    return df, semantic_time, openai_time

# ────────────────────────────────────────────────
#  MENTOR MATCHING
# ────────────────────────────────────────────────
def match_profiles_auto(cv_summary, job_interest):
    if st.session_state.df_profiles.empty:
        st.error("Profiles not available in session.")
        return pd.DataFrame(), 0, 0
    df = st.session_state.df_profiles.copy()

    # Compute profile embeddings only once
    if st.session_state.p_emb is None:
        if os.path.exists(P_EMB_PATH):
            with open(P_EMB_PATH, 'rb') as f:
                st.session_state.p_emb = pickle.load(f)
            st.success("Profile embeddings loaded from cache.")
        else:
            with st.spinner("Computing profile embeddings (done only once after embeddings reset, about 1 minute)..."):
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
                # Save to cache
                with open(P_EMB_PATH, 'wb') as f:
                    pickle.dump(st.session_state.p_emb, f)

    # Semantic search
    semantic_start = time.time()
    cv_emb = embedder.encode(cv_summary, convert_to_tensor=True)
    int_emb = embedder.encode(job_interest, convert_to_tensor=True) if job_interest else cv_emb
    q_emb = (cv_emb + int_emb)/2
    p_emb = st.session_state.p_emb

    df["match_score"] = np.round(util.cos_sim(q_emb, p_emb)[0].cpu().numpy() * 100, 2)
    df = df.sort_values("match_score", ascending=False, kind='mergesort').head(MAX_PROFILES).reset_index(drop=True)
    semantic_time = round(time.time() - semantic_start, 2)

    # OpenAI
    openai_start = time.time()
    prompts = []
    for _, row in df.iterrows():
        prompts.append(f"Summarize mentor profile in 50 words:\n{row['about']}")
        prompts.append(f"Why mentor match? 50 words:\nMy CV: {cv_summary}")
        prompts.append(f"50-word LinkedIn message to {row['name']} for 15min career chat, casual style.")
    res = generate_batch(prompts, max_tokens_per=200)

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
    openai_time = round(time.time() - openai_start, 2)
    
    return df, semantic_time, openai_time

# ────────────────────────────────────────────────
#  PAGES 
# ────────────────────────────────────────────────
def page_upload_cv():
    st.title("🌉 CareerBridge AI")
    st.header(f"Your CV and Job Interests")
    mtime = os.path.getmtime("app.py")
    dt = datetime.datetime.fromtimestamp(mtime, tz=pytz.timezone('Asia/Hong_Kong'))
    version = dt.strftime('%y%m%d_%H%M')
    st.write(f"(beta version: {version})")

    # Prepare sample CV options
    sample_dir = "sample"
    file_dict = {}
    selected = None
    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.pdf', '.docx'))]
        if files:
            file_dict = {os.path.splitext(f)[0]: f for f in files}

    col1, x, col2 = st.columns([5, 1, 5]) 

    with col1:
        st.subheader(f"Step 1. Your CV 📄")
        file = st.file_uploader("Upload your CV", type=["pdf","docx"])
        if file_dict:
            selected = st.selectbox("Or select a sample CV", sorted(list(file_dict.keys())))
    with col2:
        st.subheader(f"Step 2. Your Interests ⭐")
        st.session_state.job_interest = st.text_area("Tell me about your job interests (optional)", placeholder="e.g., Software Engineer\nSkills: Python, Machine Learning", value=st.session_state.job_interest, height=140).strip()

    st.subheader(f"Step 3. Process ⚙️")
    if st.button("Click to Process CV", type="primary", use_container_width=True):
        txt = ""
        if file:
            txt = cv_parse(file)
            #st.write(txt)
        elif selected:
            full_name = file_dict[selected]
            filepath = os.path.join(sample_dir, full_name)
            try:
                with open(filepath, 'rb') as f:
                    txt = cv_parse(f)
            except Exception as e:
                st.error(f"Error loading sample CV: {e}")
        else:
            st.error("Please upload a CV or select a sample CV.")
        
        if txt:
            st.session_state.cv_text = txt
            with st.spinner("Executing task...", show_time=True):
                start_time = time.time()
                st.session_state.cv_summary = cv_summary(txt)
                st.success(f"CV processed in {round(time.time() - start_time, 2)}s")

    if st.session_state.cv_summary:
        st.divider()
        st.subheader("📊 CV Summary")
        st.markdown(st.session_state.cv_summary)

def page_cv_suggestions():
    st.title("💡 CV Suggestions")
    st.write("Get AI-generated suggestions to improve your CV.")
    if not st.session_state.cv_text:
        st.warning("Upload and process your CV first")
        return
    if not st.session_state.cv_suggestions:
        with st.spinner("Executing task...", show_time=True):
            start_time = time.time()
            st.session_state.cv_suggestions = cv_suggestion(st.session_state.cv_text)
            st.success(f"CV suggestions processed in {round(time.time() - start_time, 2)}s")
    st.markdown(st.session_state.cv_suggestions)

def page_matched_jobs():
    st.title("🔍 Matched Jobs")
    st.write("Discover job opportunities that match your CV and interests (top 10 listed).")
    if not st.session_state.cv_summary:
        st.warning("Upload and process your CV first")
        return

    if st.session_state.df_jobs.empty:
        st.session_state.df_jobs = load_jobs_data()
    
    if st.session_state.matched_jobs.empty:
        with st.spinner("Executing task...", show_time=True):
            matched_jobs_df, semantic_time, openai_time = match_jobs_auto(
                st.session_state.cv_summary,
                st.session_state.job_interest
            )
            st.session_state.matched_jobs = matched_jobs_df
            st.success(f"Matched jobs processed - Semantic Search: {semantic_time}s | OpenAI: {openai_time}s")
    
    for _, row in st.session_state.matched_jobs.iterrows():
        with st.expander(f"**{row['position']}** - Score: {round(row['match_score'],2)}%"):
            col1, col2 = st.columns([3, 1])  
            with col1:
                st.write(f"**Company**: {row.get('company')}")
                st.write(f"**Location**: {row.get('location')}")
                st.write(f"**Summary**:  \n{row.get('summary')}")
                st.write(f"**Why Fit?**  \n{row.get('reason')}")
            
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
                    st.session_state.target_job_desc = row['description']
                    st.session_state.current_page = "target_job"
                    st.rerun()

def page_target_job():
    st.title("🎯 Target Job Tailoring")
    st.write("Enter the job description you want to tailor your CV for.")

    if not st.session_state.cv_summary:
        st.warning("Upload and process your CV first")
        return
    job_desc = st.text_area("Provide target job description or select one from matched jobs",  value=st.session_state.target_job_desc, height=200)
    if st.button("Tailor CV for Job and Suggest improvements", type="primary", use_container_width=True):
        if job_desc.strip():
            with st.spinner("Processing...", show_time=True):
                start_time = time.time()
                # Improvement 1: CV content improvements
                prompt1 = f"Based on CV summary: {st.session_state.cv_summary}, suggest specific improvements to CV content to better match this job: {job_desc}. Provide 3-5 bullet points."
                st.session_state.cv_improvements = generate_text(prompt1, max_tokens=600)
                # Improvement 2: Candidate short/long term improvements
                prompt2 = f"Based on CV summary: {st.session_state.cv_summary}, suggest improvements for the candidate to better qualify for this job: {job_desc}. Provide 3-5 bullet points."
                st.session_state.candidate_improvements = generate_text(prompt2, max_tokens=600)
                st.success(f"Processed in {round(time.time() - start_time, 2)}s")
        else:
            st.error("Please enter a job description.")
    
    if st.session_state.cv_improvements or st.session_state.candidate_improvements:
        st.divider()
        col1, x, col2 = st.columns([5, 1, 5]) 
        with col1:
            st.subheader("📄 CV Content Improvements")
            st.markdown(st.session_state.cv_improvements)
        with col2:
            st.subheader("🚀 Candidate Development Suggestions")
            st.markdown(st.session_state.candidate_improvements)

def page_matched_profiles():
    st.title("👥 Career Mentors")
    st.write("Find potential career mentors based on your profile (top 10 listed).")
    if not st.session_state.cv_summary:
        st.warning("Upload and process your CV first")
        return

    if st.session_state.df_profiles.empty:
        df = load_profiles_data()
        df = df.dropna(subset=['position','about']).query("position!='' & about!=''")
        st.session_state.df_profiles = df
    
    if st.session_state.matched_profiles.empty:
        with st.spinner("Executing task...", show_time=True):
            matched_profiles_df, semantic_time, openai_time = match_profiles_auto(
                st.session_state.cv_summary,
                st.session_state.job_interest
            )
            st.session_state.matched_profiles = matched_profiles_df
            st.success(f"Matched mentors processed - Semantic Search: {semantic_time}s | OpenAI: {openai_time}s")
    
    for _, row in st.session_state.matched_profiles.iterrows():
        with st.expander(f"**{row['name']}** - Score: {round(row['match_score'],2)}%"):
            col1, col2 = st.columns([3, 1])  
            with col1:
                st.write(f"**Position**:  \n{row.get('position')}")
                st.write(f"**Profile Summary**:  \n{row.get('summary')}")
                st.write(f"**Why Fit?**  \n{row.get('reason')}")
                #st.write(f"Debug: Position: {row.get('position')}, Reason: {row.get('reason')}")
                st.divider()
                st.markdown(f"**☕ Coffee Chat Message**  \n{row.get('greeting')}")
            
            with col2:
                if "url" in row and row['url']:
                    linkedin_url = row['url']
                    st.link_button(
                        label="View LinkedIn Profile",
                        url=linkedin_url,
                        use_container_width=True,
                        type="primary"
                    )

def page_info():
    st.title("ℹ️ About CareerBridge AI")
    
    st.header("📝 App Summary")
    st.write("""
    CareerBridge AI is a Streamlit-based application designed to assist users in their career development. It leverages AI to analyze CVs, provide personalized suggestions, match job opportunities, and connect with potential mentors. The app uses semantic search and generative AI to deliver tailored insights and recommendations. 
    """)
    
    st.header("📖 Quick User Guide")
    st.write("""
    - 📄 **Upload CV**: On the main page, upload your CV (PDF or DOCX) or select a sample CV. Optionally, enter your job interests.
    - ⚙️ **Process CV**: Click "Click to Process CV" to generate a summary.
    - 💡 **CV Suggestions**: Navigate to get AI-generated improvement suggestions for your CV.
    - 🔍 **Match Jobs**: Find job matches based on your CV and interests, with options to tailor for specific jobs.
    - 🎯 **Target Job Tailoring**: Enter a job description to receive CV improvements and development suggestions.
    - 👥 **Look for Mentors**: Discover potential career mentors and get LinkedIn message templates.
    """)
    
    st.header("🛠️ System Design Summary")
    st.write("""
    - 💻 **Frontend**: Built with Streamlit for an interactive web interface.
    - 🧠 **AI Models**: Uses Azure OpenAI for text generation and Sentence Transformers for semantic similarity matching.
    - 📊 **Data Processing**: Handles PDF and DOCX files for CV parsing, with embeddings computed for efficient job and profile matching.
    - ⚡ **Caching**: Employs Streamlit's caching for models and data, plus persistent disk caching for embeddings to optimize performance across sessions.
    - 🔄 **Session Management**: Maintains user state across pages for a seamless experience.
    """)
    
    st.header("🚀 Future Enhancements")
    st.write("""
    - 🎛️ **Filtering**: Add filters for jobs by location, expected salary, company, and other criteria to refine search results.
    - 📑 **Paging**: Implement pagination for job and mentor search results to handle large datasets efficiently.
    - 👤 **Membership**: Introduce user accounts with saved profiles, search history, and personalized recommendations.
    - 📡 **Real-Time Data**: Integrate APIs for live job postings and mentor availability updates.
    - 🧩 **Advanced Matching**: Enhance semantic search with user feedback loops and machine learning for better accuracy.
    - 🔔 **Notifications**: Add email or in-app alerts for new job matches or mentor connections.
    - 📈 **Analytics**: Provide users with insights into their CV strength and career progression.
    """)

# ────────────────────────────────────────────────
#  NAVIGATION
# ────────────────────────────────────────────────
def main():
    add_highlighted_button_css()
    with st.sidebar:
        st.title("🗺️ Navigation")
        if st.button("📄 Upload CV", use_container_width=True):
            st.session_state.current_page = "upload_cv"
        if st.button("💡 CV Suggestions", use_container_width=True):
            st.session_state.current_page = "cv_suggestions"
        if st.button("🔍 Match Jobs", use_container_width=True):
            st.session_state.current_page = "matched_jobs"
        if st.button("🎯 Target Job Tailoring", use_container_width=True):
            st.session_state.current_page = "target_job"
        if st.button("👥 Look for Mentors", use_container_width=True):
            st.session_state.current_page = "matched_profiles"
        if st.button("ℹ️ Info", use_container_width=True):
            st.session_state.current_page = "info"
        
        st.divider()
        if st.button("🔄 Reset Embeddings", use_container_width=True):
            if os.path.exists(J_EMB_PATH):
                os.remove(J_EMB_PATH)
            if os.path.exists(P_EMB_PATH):
                os.remove(P_EMB_PATH)
            st.success("Embeddings cache reset.")

    pages = {
        "upload_cv": page_upload_cv,
        "cv_suggestions": page_cv_suggestions,
        "target_job": page_target_job,
        "matched_jobs": page_matched_jobs,
        "matched_profiles": page_matched_profiles,
        "info": page_info
    }
    pages[st.session_state.current_page]()

if __name__ == "__main__":
    main()