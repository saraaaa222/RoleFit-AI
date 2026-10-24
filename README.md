# RoleFit AI — Career Intelligence & Skill-Gap Analyzer

RoleFit AI is an interactive career exploration application that helps
users discover relevant job opportunities, understand skill overlaps,
and identify potential areas for development.

## Features

- Career-direction-aware job recommendations
- TF-IDF text representation and cosine similarity
- Heuristic ranking combining text relevance, skill overlap, and
  selected career direction
- Dictionary-based skill matching and potential skill-gap analysis
- Filters for location, remote work, and employment type
- Job descriptions and original posting links where available
- Interactive dashboard built with Streamlit

## Technology Stack

Python · Streamlit · Pandas · Scikit-learn · NLP · TF-IDF

## Architecture

1. Load and validate the job dataset.
2. Clean job descriptions and prepare searchable text.
3. Transform job text using TF-IDF.
4. Compare the user's profile with job records using cosine similarity.
5. Extract recognised skills and calculate overlap.
6. Rank results using relevance signals and selected career direction.
7. Display recommendations, skill gaps, and job details in Streamlit.

## Run Locally

Python 3.10 or newer is recommended.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run rolefit_app.py
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Dataset

The application reads `jobs_dataset.csv` from the project directory.
The dataset contains job titles, descriptions, company information,
locations, job types, and other job-posting metadata.

Dataset source:
https://www.kaggle.com/datasets/promptcloud/indeed-india-job-dataset

Check the dataset's applicable terms before redistributing it.
The application may require the dataset to be supplied separately
when deploying to a hosting platform.

## Important Limitations

- Match scores are heuristic relevance indicators, not hiring probabilities.
- Skill extraction uses a predefined dictionary and can miss synonyms
  or skills that are expressed differently.
- A detected skill is not proof of proficiency.
- Job records may be outdated; verify details on the original posting.
- Career-family and seniority adjustments are heuristics, not a verified
  assessment of candidate eligibility.
- Recommendations depend on the quality and coverage of the dataset.

## Validation

Run the project validation script:

```bash
python validate_project.py
```

## Screenshots

Add screenshots of the final, tested application to a `screenshots/`
directory and link them here before publication.

## Project Status

Portfolio prototype. Features, tests, deployment status, and results
should be described according to what has actually been implemented
and verified.

## Acknowledgements

Built with Streamlit, Pandas, and Scikit-learn.
Dataset source: PromptCloud's Indeed India job dataset on Kaggle.

