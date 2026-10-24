# RoleFit AI — Career Intelligence & Skill-Gap Analyzer

**Discover relevant opportunities. Understand your skill gaps. Explore your next career move.**

RoleFit AI is an interactive AI-powered career exploration dashboard that matches a user's skills, experience, and career interests with job opportunities. It combines semantic text similarity, keyword-based matching, skill overlap, and career-direction signals to generate relevant recommendations.

**[Launch the Live Demo](https://rolefit-ai-ncvdbyslwt2t4px6tdfsk.streamlit.app)** · **[View the Source Code](https://github.com/saraaaa222/RoleFit-AI)**

## Features

- **Two matching modes:** Compare Transformer-based semantic matching with TF-IDF keyword matching.
- **Career-aware recommendations:** Explore opportunities across AI/ML, Data Science, Data Analytics, Business Analytics, and Product & AI Strategy.
- **Skill-gap insights:** Identify recognised matching skills and potential skills to develop.
- **Personalised filtering:** Narrow results by location, remote-work preference, employment type, and other available filters.
- **Early-career focus:** Prioritise entry-level opportunities using heuristic ranking adjustments.
- **Job-level details:** Review job descriptions, available metadata, and original posting links.
- **Interactive dashboard:** Explore recommendations through a Streamlit interface.

## How It Works

1. **Load the dataset:** Read job records and relevant metadata from `jobs_dataset.csv`.
2. **Prepare job text:** Process available job titles and descriptions into searchable representations.
3. **Match the profile:** Use either Transformer embeddings for semantic similarity or TF-IDF vectors for keyword-based similarity.
4. **Analyse skills:** Compare recognised profile skills with skills detected in job descriptions.
5. **Rank opportunities:** Combine relevance, skill overlap, and career-direction signals, with additional heuristic adjustments where applicable.
6. **Present insights:** Display ranked opportunities, matching skills, potential gaps, and job details.

### Matching Methods

| Method | Approach |
|---|---|
| Transformer semantic matching | Uses `sentence-transformers/all-MiniLM-L6-v2` to represent text as embeddings and compare semantic similarity. |
| TF-IDF keyword matching | Represents text using term importance and compares vectors using cosine similarity. |

The two methods offer different ways to compare a candidate profile with job descriptions. Their results depend on the profile, dataset, and matching approach.

## Technology Stack

- **Language:** Python
- **Application:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning and similarity:** Scikit-learn
- **Natural language processing:** TF-IDF, Sentence Transformers
- **Semantic model:** `all-MiniLM-L6-v2`
- **Deployment:** Streamlit Community Cloud

## Run Locally

### Prerequisites

- Python 3.10 or newer
- Git
- The project dataset, subject to its applicable terms

### Installation

Clone the repository:

```bash
git clone https://github.com/saraaaa222/RoleFit-AI.git
cd RoleFit-AI
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it using:

```bash
.venv\Scripts\activate
```

Install the dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run rolefit_app.py
```

Open the local URL displayed in your terminal.

### Dataset Setup

The application expects `jobs_dataset.csv` in the project directory.

Dataset source: [PromptCloud — Indeed India Job Dataset on Kaggle](https://www.kaggle.com/datasets/promptcloud/indeed-india-job-dataset)

Check the dataset's licence and applicable terms before using or redistributing it. If you obtain the dataset separately, place the CSV in the project directory before running the application.

## Validation

Run the project's validation script:

```bash
python validate_project.py
```

You can also check Python syntax with:

```bash
python -m py_compile rolefit_app.py semantic_engine.py validate_project.py
```

## Screenshots

Add screenshots of the working application to a `screenshots/` directory to showcase the profile inputs, matching options, filters, and ranked recommendations.

## Important Limitations

- **Relevance scores are not probabilities:** They do not represent the probability of being hired
