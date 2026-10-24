import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_engine import encode_job_texts, rank_by_semantic_similarity

st.set_page_config(
    page_title="RoleFit AI | Career Intelligence",
    page_icon="🎯",
    layout="wide",
)

SKILLS = [
    "python", "java", "c++", "sql", "excel", "power bi", "tableau",
    "machine learning", "deep learning", "artificial intelligence",
    "data analysis", "data analytics", "statistics", "linear algebra",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "natural language processing", "nlp", "computer vision",
    "generative ai", "llm", "rag", "prompt engineering",
    "data visualization", "data visualisation", "matplotlib",
    "seaborn", "aws", "azure", "gcp", "docker", "git", "github",
    "fastapi", "flask", "streamlit", "postgresql", "mysql",
    "mongodb", "api", "html", "css", "javascript", "react",
    "project management", "communication", "leadership",
    "problem solving", "business analysis", "market research",
    "product management", "agile", "scrum", "financial analysis",
    "forecasting", "regression", "classification", "data cleaning",
    "etl", "big data", "spark", "hadoop", "linux",
]


def clean_text(value):
    if pd.isna(value):
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_skills(text):
    text = str(text).lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])"
        if re.search(pattern, text):
            found.append(skill)
    return sorted(set(found))


@st.cache_data(show_spinner="Loading job dataset...")
def load_jobs():
    path = Path(__file__).parent / "jobs_dataset.csv"
    data = pd.read_csv(path)

    required = ["job_title", "job_description"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Dataset is missing required column: {col}")

    for col in [
        "company_name", "city", "state", "country", "job_type",
        "url", "is_remote", "has_expired", "category"
    ]:
        if col not in data.columns:
            data[col] = ""

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna("")

    data["job_title"] = data["job_title"].map(clean_text)
    data["job_description"] = data["job_description"].map(clean_text)
    data["search_text"] = (
        data["job_title"] + " " + data["job_description"]
    ).str.lower()

    data = data[data["job_title"].str.len() > 0].copy()
    return data.reset_index(drop=True)


@st.cache_resource(show_spinner="Preparing matching engine...")
def build_index(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def safe_value(value, fallback="Not specified"):
    value = clean_text(value)
    return value if value else fallback




st.markdown("""
<style>
:root {
    --ivory: #FAF6F2;
    --rose: #E8CDD4;
    --rose-light: #F3E5E8;
    --burgundy: #783F52;
    --burgundy-hover: #603142;
    --text: #38252B;
    --muted: #75666B;
    --border: #E7D8DC;
    --white: #FFFFFF;
}

.stApp {
    background: var(--ivory);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(250, 246, 242, 0.96);
}

[data-testid="stSidebar"] {
    background: var(--rose-light);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

h1, h2, h3, p, label {
    color: var(--text);
}

.hero {
    padding: 2rem 2.2rem;
    border: 1px solid #DDBFC8;
    border-radius: 20px;
    background: linear-gradient(125deg, #E8CDD4 0%, #F3E5E8 100%);
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(91, 48, 64, 0.07);
}

.hero h1 {
    color: var(--text);
    font-size: 2.5rem;
    font-weight: 750;
}

.hero p {
    color: #513B43;
    font-size: 1.05rem;
    line-height: 1.65;
}

.muted {
    color: #604750;
    font-size: 0.84rem;
    font-weight: 600;
    letter-spacing: 1.1px;
}

.job-card {
    padding: 1.35rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    margin: 0.85rem 0;
    background: var(--white);
    box-shadow: 0 5px 18px rgba(65, 37, 46, 0.05);
}

.score {
    font-size: 1.55rem;
    font-weight: 750;
    color: var(--burgundy);
}

div.stButton > button,
div.stDownloadButton > button {
    background: var(--burgundy);
    color: var(--white);
    border: 1px solid var(--burgundy);
    border-radius: 10px;
    font-weight: 600;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: var(--burgundy-hover);
    color: var(--white);
    border-color: var(--burgundy-hover);
}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: var(--white) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    border-radius: 8px;
}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    background: var(--white) !important;
    color: var(--text) !important;
    border-color: #D7BEC5 !important;
    border-radius: 9px;
}

div[data-baseweb="select"] *,
div[data-baseweb="popover"] *,
ul[role="listbox"] * {
    color: var(--text) !important;
}

div[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 14px;
}

input, textarea {
    caret-color: var(--burgundy);
}

hr {
    border-color: var(--border);
}

a {
    color: var(--burgundy);
}

@media (max-width: 768px) {
    .hero {
        padding: 1.4rem;
    }
    .hero h1 {
        font-size: 2rem;
    }
    .job-card {
        padding: 1rem;
    }
}
/* Force readable light form controls */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="textarea"] > div {
    background-color: #FFFFFF !important;
    color: #38252B !important;
    border-color: #D7BEC5 !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    -webkit-text-fill-color: #38252B !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="popover"] *,
[data-testid="stSidebar"] [role="listbox"] * {
    color: #38252B !important;
}

[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #75666B !important;
    -webkit-text-fill-color: #75666B !important;
}

/* Replace bright red accents with burgundy */
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child,
[data-testid="stSidebar"] [data-baseweb="checkbox"] > div {
    accent-color: #783F52 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] {
    accent-color: #783F52 !important;
}

/* Pink controls with black text */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #E8CDD4 !important;
    border-color: #D5AEBB !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}

[data-testid="stSidebar"] [data-baseweb="popover"],
[data-testid="stSidebar"] [role="listbox"] {
    background-color: #F3E5E8 !important;
}

[data-testid="stSidebar"] [role="option"] {
    background-color: #F3E5E8 !important;
    color: #000000 !important;
}

[data-testid="stSidebar"] [role="option"]:hover {
    background-color: #E8CDD4 !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: #783F52 !important;
    border-color: #783F52 !important;
}

[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #000000 !important;
}

[data-testid="stSidebar"] [data-testid="stCheckbox"] input,
[data-testid="stSidebar"] [role="radio"] {
    accent-color: #783F52 !important;
}
/* Fix dark dropdown shells */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #E8CDD4 !important;
    border: 1px solid #D5AEBB !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] input {
    background: transparent !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #000000 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}

/* Fix job posting buttons */
.stApp div.stButton > button,
.stApp div.stLinkButton > a,
.stApp div[data-testid="stLinkButton"] a {
    background: #E8CDD4 !important;
    color: #000000 !important;
    border: 1px solid #D5AEBB !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

.stApp div.stButton > button:hover,
.stApp div.stLinkButton > a:hover,
.stApp div[data-testid="stLinkButton"] a:hover {
    background: #DDBFC8 !important;
    color: #000000 !important;
    border-color: #B994A3 !important;
}

/* Make slider accents match the theme */
.stApp [data-testid="stSlider"] [role="slider"] {
    background: #783F52 !important;
    border-color: #783F52 !important;
}

/* Keep checkbox labels readable */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #000000 !important;
}


</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="hero">
  <div class="muted">CAREER INTELLIGENCE · PROJECT 01</div>
  <h1>RoleFit AI</h1>
  <p>Discover relevant roles, understand your skill gaps, and make
  your next career move with evidence-based matching.</p>
</div>
""", unsafe_allow_html=True)

try:
    jobs = load_jobs()
except Exception as exc:
    st.error(f"Could not load jobs_dataset.csv: {exc}")
    st.stop()

with st.sidebar:
    st.header("Your career profile")
    st.caption("Enter skills you genuinely have. You can edit this profile anytime.")

    skills_input = st.text_area(
        "Your skills",
        value="Python, SQL, machine learning, statistics, data analysis",
        height=110,
        help="Separate skills with commas.",
    )
    interests_input = st.text_input(
        "Career interests",
        placeholder="e.g. AI, analytics, product",
    )
    career_family = st.selectbox(
        "Target career direction",
        [
            "Any relevant role",
            "AI & Machine Learning",
            "Data Science",
            "Data Analytics",
            "Business Analytics",
            "Product & AI Strategy",
        ],
    )
    downrank_senior = st.checkbox(
        "Prioritise early-career opportunities",
        value=True,
        help="Down-ranks clearly senior or executive titles; it cannot verify experience requirements.",
    )
    location_input = st.text_input(
        "Preferred location",
        placeholder="e.g. Mumbai",
    )
    remote_only = st.checkbox("Remote roles only")
    job_type_options = sorted(
        x for x in jobs["job_type"].astype(str).unique()
        if x.strip()
    )
    selected_types = st.multiselect(
        "Employment type",
        options=job_type_options,
        default=[],
    )
    result_count = st.slider("Number of roles", 5, 20, 10)

profile_text = " ".join([skills_input, interests_input]).strip()
profile_skills = extract_skills(profile_text)

if not profile_text:
    st.info("Add skills or career interests in the sidebar to begin.")
    st.stop()

if not profile_skills:
    st.warning(
        "No skills from the current skill dictionary were detected. "
        "Your text can still be used for matching, but skill-gap analysis "
        "will be limited. Try adding specific skills such as Python or SQL."
    )

matching_method = st.radio(
    "Matching engine",
    ["Transformer semantic matching", "TF-IDF keyword matching"],
    horizontal=True,
    help="Transformer matching uses sentence embeddings; TF-IDF is a lightweight keyword-based baseline.",
)

try:
    job_texts = tuple(jobs["search_text"].fillna("").astype(str).tolist())

    if matching_method == "Transformer semantic matching":
        @st.cache_data(show_spinner="Encoding job descriptions with the Transformer model...")
        def cached_job_embeddings(texts):
            return encode_job_texts(list(texts))

        job_embeddings = cached_job_embeddings(job_texts)
        semantic_scores = rank_by_semantic_similarity(
            profile_text,
            job_embeddings,
        )
    else:
        vectorizer, job_matrix = build_index(job_texts)
        query_vector = vectorizer.transform([profile_text.lower()])
        semantic_scores = cosine_similarity(query_vector, job_matrix).ravel()

except Exception as exc:
    st.error(f"Could not prepare the recommendation engine: {exc}")
    st.stop()

results = jobs.copy()
results["semantic_score"] = semantic_scores
results["job_skills"] = results["search_text"].map(extract_skills)
results["matched_skills"] = results["job_skills"].map(
    lambda items: sorted(set(items).intersection(profile_skills))
)
results["missing_skills"] = results["job_skills"].map(
    lambda items: sorted(set(items).difference(profile_skills))
)

CAREER_KEYWORDS = {
    "AI & Machine Learning": [
        "machine learning", "artificial intelligence", "ai engineer",
        "ml engineer", "deep learning", "nlp", "computer vision",
        "data scientist", "research scientist",
    ],
    "Data Science": [
        "data scientist", "data science", "statistician",
        "quantitative analyst", "research analyst",
    ],
    "Data Analytics": [
        "data analyst", "data analytics", "analytics",
        "business intelligence", "bi analyst", "reporting analyst",
    ],
    "Business Analytics": [
        "business analyst", "business analytics",
        "operations analyst", "strategy analyst",
    ],
    "Product & AI Strategy": [
        "product manager", "product analyst", "product owner",
        "ai product", "product strategy",
    ],
}

SENIOR_TERMS = [
    "senior", "sr.", "sr ", "lead", "principal", "director",
    "vice president", "vp ", "head of", "chief", "executive",
]

def explainable_score(row):
    job_skills = set(row["job_skills"])
    profile_set = set(profile_skills)
    overlap = len(profile_set.intersection(job_skills))
    coverage = overlap / max(len(job_skills), 1)

    title = str(row["job_title"]).lower()
    category = str(row.get("category", "")).lower()
    description = str(row["job_description"]).lower()

    if career_family == "Any relevant role":
        family_score = 0.5
    else:
        keywords = CAREER_KEYWORDS[career_family]
        if any(keyword in title for keyword in keywords):
            family_score = 1.0
        elif any(keyword in category for keyword in keywords):
            family_score = 0.7
        elif any(keyword in description for keyword in keywords):
            family_score = 0.35
        else:
            family_score = 0.0

    # A relevance indicator, not a hiring probability.
    score = (
        0.40 * float(row["semantic_score"])
        + 0.30 * coverage
        + 0.30 * family_score
    ) * 100

    if downrank_senior:
        senior_terms = [
            "senior", "sr.", "sr ", "lead", "principal",
            "director", "vice president", "vp ", "head of",
            "chief", "executive", "general manager",
        ]
        early_terms = [
            "intern", "internship", "fresher", "trainee",
            "junior", "entry level", "entry-level",
            "graduate", "apprentice",
        ]

        job_type_text = str(row.get("job_type", "")).lower()
        is_senior = any(term in title for term in senior_terms)
        is_early = (
            any(term in title for term in early_terms)
            or any(term in job_type_text for term in early_terms)
        )

        if is_senior:
            score -= 25
        if is_early:
            score += 8

    return round(max(0.0, min(100.0, score)), 1)

results["match_score"] = results.apply(explainable_score, axis=1)

if location_input.strip():
    loc = location_input.strip()
    location_mask = (
        results["city"].astype(str).str.contains(loc, case=False, na=False)
        | results["state"].astype(str).str.contains(loc, case=False, na=False)
        | results["country"].astype(str).str.contains(loc, case=False, na=False)
    )
    results = results[location_mask]

if remote_only:
    remote_mask = results["is_remote"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    results = results[remote_mask]

if selected_types:
    results = results[
        results["job_type"].astype(str).isin(selected_types)
    ]

results = results.sort_values(
    ["match_score", "semantic_score"], ascending=False
).head(result_count)

st.subheader("Your career snapshot")
m1, m2, m3 = st.columns(3)
m1.metric("Jobs in dataset", f"{len(jobs):,}")
m2.metric("Roles matching filters", f"{len(results):,}")
m3.metric("Skills detected", len(profile_skills))

if profile_skills:
    st.caption("Detected skills: " + " · ".join(profile_skills))

st.divider()
st.subheader("Recommended roles")
st.caption(
    "Match scores are heuristic relevance indicators based on text similarity "
    "and skill overlap. They are not hiring probabilities or verified suitability scores."
)

if results.empty:
    st.warning("No jobs match these filters. Try clearing the location or employment filters.")
else:
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        title = safe_value(row["job_title"])
        company = safe_value(row["company_name"])
        city = safe_value(row["city"], "")
        state = safe_value(row["state"], "")
        location = ", ".join(x for x in [city, state] if x and x != "Not specified")
        if not location:
            location = "Location not specified"

        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"### {rank}. {title}")
                st.write(f"**{company}** · {location}")
                st.caption(
                    f"Type: {safe_value(row['job_type'])} · "
                    f"Remote: {safe_value(row['is_remote'])}"
                )
            with right:
                st.metric("Match indicator", f"{row['match_score']:.1f}/100")

            matched = row["matched_skills"]
            missing = row["missing_skills"]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Skills found in your profile**")
                st.write(", ".join(matched) if matched else "No dictionary skills matched")
            with c2:
                st.markdown("**Potential skills to develop**")
                st.write(", ".join(missing[:12]) if missing else "No additional dictionary skills detected")

            with st.expander("Read job description and matching details"):
                st.write(safe_value(row["job_description"], "No description available."))
                st.caption(
                    f"Text similarity: {row['semantic_score']:.3f} · "
                    f"Detected job skills: {len(row['job_skills'])}"
                )

            url = safe_value(row["url"], "")
            if url.startswith(("https://", "http://")):
                st.link_button("View original posting", url)
            else:
                st.caption("No valid application URL available.")

st.divider()
st.subheader("Your skill development priorities")

if profile_skills and not results.empty:
    gap_counts = {}
    for missing in results["missing_skills"]:
        for skill in missing:
            gap_counts[skill] = gap_counts.get(skill, 0) + 1

    if gap_counts:
        gap_df = pd.DataFrame(
            [{"Skill": skill, "Roles mentioning skill": count}
             for skill, count in gap_counts.items()]
        ).sort_values("Roles mentioning skill", ascending=False).head(10)

        st.bar_chart(gap_df.set_index("Skill"))
        st.caption(
            "These are skills detected in the selected job descriptions "
            "that are not in your entered profile. They are not guaranteed "
            "requirements; review each job description before prioritising."
        )
    else:
        st.info("No additional skills from the current dictionary were detected in these roles.")
else:
    st.info("Add recognisable skills and generate matching roles to see skill priorities.")

st.caption(
    "RoleFit AI is a portfolio prototype. Job data may be outdated, "
    "skill extraction is dictionary-based, and recommendations should be "
    "verified against original job postings."
)
