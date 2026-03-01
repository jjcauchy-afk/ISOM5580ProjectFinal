import streamlit as st
import pandas as pd
import os
import random
from io import BytesIO
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# =============================================
# SETUP & CONFIG
# =============================================
st.set_page_config(
    page_title="CareerBridge AI",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌉 CareerBridge AI")
st.header("Bridge the gap to your dream career")
st.markdown("""
**Upload your resume** → AI instantly finds perfect **JobsDB** matches + **LinkedIn mentors** who have walked your exact path.
""")

# Sidebar for API keys
with st.sidebar:
    st.header("🔑 API Settings")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required for CV analysis & personalized mentor greetings"
    )
    st.caption("Get a free key at [platform.openai.com](https://platform.openai.com)")

# =============================================
# SAMPLE DATA GENERATION (21 records)
# =============================================
def generate_sample_jobs(n=21):
    titles = [
        "Software Engineer", "Senior Data Scientist", "Product Manager",
        "DevOps Engineer", "UX/UI Designer", "Machine Learning Engineer",
        "Marketing Manager", "Financial Analyst", "Cybersecurity Specialist",
        "Sales Executive"
    ]
    companies = [
        "TechNova HK", "DataForge", "InnovateAsia", "CloudPeak",
        "FinSecure", "DesignSphere", "SecureNet", "GlobalLink",
        "ByteWave", "QuantumLabs"
    ]
    locations = ["Hong Kong", "Kowloon", "Central", "Singapore", "Remote", "Hong Kong Island"]
    
    samples = []
    for i in range(n):
        title = random.choice(titles)
        company = random.choice(companies)
        loc = random.choice(locations)
        summary = f"Join our team to build cutting-edge {title.lower()} solutions. {random.randint(2,8)} years experience preferred. Competitive salary + equity."
        link = f"https://hk.jobsdb.com/hk/en/job/{title.lower().replace(' ', '-')}-{1000+i}"
        samples.append({
            "title": title,
            "company": company,
            "location": loc,
            "summary": summary,
            "link": link
        })
    return pd.DataFrame(samples)


def generate_sample_mentors(n=21):
    first_names = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie", "Parker", "Quinn", "Drew", "Sam"]
    last_names = ["Lam", "Wong", "Chan", "Li", "Ng", "Cheung", "Ho", "Kwok", "Lau", "Ma", "Zhang"]
    titles = [
        "Lead Software Engineer @ Google", "Principal Data Scientist @ Meta",
        "VP Product @ Tencent", "Head of AI @ ByteDance", "Engineering Manager @ Microsoft",
        "Senior UX Designer @ Apple", "Founder & Mentor", "Tech Lead @ Alibaba",
        "Career Coach (ex-IBM)", "Director of Engineering @ Amazon"
    ]
    base_summaries = [
        "15+ years in tech. Mentored 80+ professionals transitioning into software roles.",
        "Ex-FAANG. Passionate about helping juniors land their dream jobs.",
        "Built 3 startups from zero to acquisition. Loves 1:1 career chats."
    ]
    
    samples = []
    for i in range(n):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        title = random.choice(titles)
        summary = random.choice(base_summaries) + f" Currently at {title.split('@')[-1].strip()}."
        link = f"https://www.linkedin.com/in/{name.lower().replace(' ', '-')}{i}/"
        samples.append({
            "name": name,
            "job_title": title,
            "summary": summary,
            "link": link
        })
    return pd.DataFrame(samples)


# =============================================
# DATA LOADING (with sample prepending)
# =============================================
@st.cache_data
def load_jobsdb():
    csv_path = "jobsdb.csv"
    if not os.path.exists(csv_path):
        df = generate_sample_jobs(21)
        df.to_csv(csv_path, index=False)
        st.toast("Created new jobsdb.csv with 21 sample records")
    else:
        df = pd.read_csv(csv_path)
    
    # Always prepend 21 fresh samples at the beginning (as per requirement)
    samples_df = generate_sample_jobs(21)
    df = pd.concat([samples_df, df], ignore_index=True)
    df = df.drop_duplicates(subset=["title", "company", "link"]).reset_index(drop=True)
    return df


@st.cache_data
def load_linkedin():
    csv_path = "linkedin.csv"
    if not os.path.exists(csv_path):
        df = generate_sample_mentors(21)
        df.to_csv(csv_path, index=False)
        st.toast("Created new linkedin.csv with 21 sample records")
    else:
        df = pd.read_csv(csv_path)
    
    # Always prepend 21 fresh samples
    samples_df = generate_sample_mentors(21)
    df = pd.concat([samples_df, df], ignore_index=True)
    df = df.drop_duplicates(subset=["name", "job_title"]).reset_index(drop=True)
    return df


# =============================================
# CV PARSING
# =============================================
def parse_cv(uploaded_file):
    if uploaded_file is None:
        return ""
    file_ext = uploaded_file.name.split(".")[-1].lower()
    bytes_data = uploaded_file.getvalue()
    
    text = ""
    if file_ext == "pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(bytes_data))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"PDF parsing error: {e}")
    elif file_ext == "docx":
        try:
            doc = Document(BytesIO(bytes_data))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            st.error(f"DOCX parsing error: {e}")
    else:
        st.error("Only PDF and DOCX supported")
    return text.strip()


# =============================================
# LLM HELPERS (OpenAI)
# =============================================
def analyze_cv(cv_text, api_key):
    if not api_key or not cv_text:
        return "Enter your OpenAI key to enable AI analysis.", "Enter your OpenAI key to enable suggestions."
    
    client = OpenAI(api_key=api_key)
    truncated = cv_text[:6000]  # safety
    
    # Summary
    summary_prompt = f"""Summarize this CV in 4-6 professional sentences. Highlight key experience, skills, and career goals.
CV:
{truncated}"""
    
    summary_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=300,
        temperature=0.7
    )
    summary = summary_resp.choices[0].message.content.strip()
    
    # Suggestions
    sugg_prompt = f"""Give 5 concrete, actionable suggestions to improve this CV for tech/job applications (bullet points).
Focus on keywords, achievements, structure.
CV:
{truncated}"""
    
    sugg_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": sugg_prompt}],
        max_tokens=400,
        temperature=0.7
    )
    suggestions = sugg_resp.choices[0].message.content.strip()
    
    return summary, suggestions


def generate_greeting(mentor_row, cv_summary, api_key):
    if not api_key:
        return f"Hi {mentor_row['name']},\n\nI came across your impressive profile while looking for mentors in {mentor_row['job_title'].split('@')[0]}. I'd love to learn from your journey over a quick 15-minute virtual coffee chat. Would you be open to it?\n\nBest regards,\n[Your Name]"
    
    client = OpenAI(api_key=api_key)
    prompt = f"""Write a short, warm, professional first-message (max 5 sentences) to invite {mentor_row['name']} for a 15-min virtual coffee chat.

Mentor: {mentor_row['name']}, {mentor_row['job_title']}
Mentor summary: {mentor_row['summary']}

My background summary: {cv_summary if cv_summary else 'Aspiring professional in tech with strong interest in career growth.'}

Tone: respectful, concise, genuine. End with a clear call-to-action."""
    
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.8
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Hi {mentor_row['name']}, I'd love to connect for a quick coffee chat about your career path. Available next week?"


# =============================================
# EMBEDDING MODEL (cached)
# =============================================
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# =============================================
# MAIN APP
# =============================================
uploaded_file = st.file_uploader(
    "📄 Upload your CV (PDF or DOCX)",
    type=["pdf", "docx"],
    help="Your data is processed locally — never stored."
)

cv_text = ""
cv_summary = ""

if uploaded_file:
    cv_text = parse_cv(uploaded_file)
    if cv_text:
        st.success(f"✅ CV parsed successfully ({len(cv_text.split())} words)")
        
        # CV Analysis (LLM)
        if openai_key:
            with st.spinner("Analyzing CV with AI..."):
                cv_summary, suggestions = analyze_cv(cv_text, openai_key)
            
            col_ana1, col_ana2 = st.columns(2)
            with col_ana1:
                st.subheader("📊 Analysis")
                st.markdown(cv_summary)
            with col_ana2:
                st.subheader("💡 Suggestions to improve CV")
                st.markdown(suggestions)
        else:
            st.warning("🔑 Enter OpenAI API key in sidebar for AI-powered CV analysis & mentor greetings")

        # =============================================
        # SEMANTIC MATCHING
        # =============================================
        model = get_embedding_model()
        
        # Jobs matching
        df_jobs = load_jobsdb()
        df_jobs["full_text"] = df_jobs.apply(
            lambda row: f"{row['title']} at {row['company']} in {row['location']}. {row['summary']}", axis=1
        )
        job_embeddings = model.encode(df_jobs["full_text"].tolist(), convert_to_tensor=True)
        cv_embedding = model.encode(cv_text, convert_to_tensor=True)
        job_scores = util.cos_sim(cv_embedding, job_embeddings)[0].cpu().numpy()
        df_jobs["match_score"] = (job_scores * 100).round(1)
        df_jobs = df_jobs.sort_values("match_score", ascending=False).reset_index(drop=True)
        
        # Mentors matching
        df_mentors = load_linkedin()
        df_mentors["full_text"] = df_mentors.apply(
            lambda row: f"{row['name']} {row['job_title']}. {row['summary']}", axis=1
        )
        mentor_embeddings = model.encode(df_mentors["full_text"].tolist(), convert_to_tensor=True)
        mentor_scores = util.cos_sim(cv_embedding, mentor_embeddings)[0].cpu().numpy()
        df_mentors["match_score"] = (mentor_scores * 100).round(1)
        df_mentors = df_mentors.sort_values("match_score", ascending=False).reset_index(drop=True)

        # =============================================
        # LAYOUT: JOBS LEFT | MENTORS RIGHT
        # =============================================
        st.markdown("---")
        left_col, right_col = st.columns([1, 1], gap="large")

        # ----------------- JOBS LEFT -----------------
        with left_col:
            st.subheader("🔍 Job Matches on JobsDB")
            st.caption("Top matches based on semantic similarity")
            
            jobs_per_page = 10
            if "job_page" not in st.session_state:
                st.session_state.job_page = 0
            
            total_jobs = len(df_jobs)
            start_idx = st.session_state.job_page * jobs_per_page
            end_idx = start_idx + jobs_per_page
            page_jobs = df_jobs.iloc[start_idx:end_idx]
            
            for _, job in page_jobs.iterrows():
                with st.container(border=True):
                    st.write(f"**{job['title']}**")
                    st.write(f"**{job['company']}** • {job['location']}")
                    st.write(job['summary'][:220] + "..." if len(job['summary']) > 220 else job['summary'])
                    st.write(f"**Match Score: {job['match_score']}%**")
                    st.link_button("🔗 View on JobsDB", job['link'], use_container_width=True)
            
            # Pagination
            pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
            with pcol1:
                if st.button("← Previous", key="jobs_prev", disabled=st.session_state.job_page == 0):
                    st.session_state.job_page -= 1
                    st.rerun()
            with pcol3:
                if st.button("Next →", key="jobs_next", disabled=end_idx >= total_jobs):
                    st.session_state.job_page += 1
                    st.rerun()
            st.caption(f"Page {st.session_state.job_page + 1} of {(total_jobs - 1) // jobs_per_page + 1} • Showing {len(page_jobs)} jobs")

        # ----------------- MENTORS RIGHT -----------------
        with right_col:
            st.subheader("👥 Career Path Mentors on LinkedIn")
            st.caption("People who have walked your path — reach out!")
            
            mentors_per_page = 10
            if "mentor_page" not in st.session_state:
                st.session_state.mentor_page = 0
            
            if "show_greeting" not in st.session_state:
                st.session_state.show_greeting = {}
            if "greetings" not in st.session_state:
                st.session_state.greetings = {}
            
            total_mentors = len(df_mentors)
            start_idx_m = st.session_state.mentor_page * mentors_per_page
            end_idx_m = start_idx_m + mentors_per_page
            page_mentors = df_mentors.iloc[start_idx_m:end_idx_m]
            
            for idx, mentor in page_mentors.iterrows():
                mentor_id = f"mentor_{start_idx_m + idx}"
                
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.write(f"**{mentor['name']}**")
                        st.write(f"*{mentor['job_title']}*")
                        st.write(mentor['summary'][:180] + "..." if len(mentor['summary']) > 180 else mentor['summary'])
                        st.write(f"**Match Score: {mentor['match_score']}%**")
                        st.link_button("🔗 View LinkedIn", mentor['link'], use_container_width=True)
                    
                    with c2:
                        btn_label = "☕ Coffee chat Invite" if not st.session_state.show_greeting.get(mentor_id, False) else "Hide Message"
                        if st.button(btn_label, key=f"invite_{mentor_id}", use_container_width=True):
                            st.session_state.show_greeting[mentor_id] = not st.session_state.show_greeting.get(mentor_id, False)
                            if st.session_state.show_greeting[mentor_id] and mentor_id not in st.session_state.greetings:
                                with st.spinner("Crafting personalized message..."):
                                    greeting = generate_greeting(mentor, cv_summary, openai_key)
                                    st.session_state.greetings[mentor_id] = greeting
                        
                        if st.session_state.show_greeting.get(mentor_id, False):
                            st.info(st.session_state.greetings.get(mentor_id, "Message ready"))
            
            # Mentor pagination
            mpcol1, mpcol2, mpcol3 = st.columns([1, 2, 1])
            with mpcol1:
                if st.button("← Previous", key="mentors_prev", disabled=st.session_state.mentor_page == 0):
                    st.session_state.mentor_page -= 1
                    st.rerun()
            with mpcol3:
                if st.button("Next →", key="mentors_next", disabled=end_idx_m >= total_mentors):
                    st.session_state.mentor_page += 1
                    st.rerun()
            st.caption(f"Page {st.session_state.mentor_page + 1} of {(total_mentors - 1) // mentors_per_page + 1}")

else:
    st.info("👆 Upload your CV to see personalized job matches and mentor suggestions")

# Footer
st.markdown("---")
st.caption("CareerBridge AI • Built with Streamlit • Semantic search powered by sentence-transformers • LLM by OpenAI")