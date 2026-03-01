# AI Resume Screener & Job Matcher

Production-grade AI application for resume-to-job matching with explainability, ATS analytics, optional LLM coaching, batch ranking, API backend, evaluation harness, and CI-tested engineering workflows.

## Problem
Manual resume screening is slow, inconsistent, and hard to justify. Recruiters and candidates need:
- A reliable alignment score (not just keyword matching)
- Clear evidence of why a score is high/low
- Actionable improvement steps
- ATS readiness insights
- Reproducible metrics to trust model behavior

## Solution
This project provides:
- Hybrid scoring (semantic embeddings + lexical overlap + skill coverage + optional cross-encoder reranking)
- Skill-gap analysis (matching, missing, extra skills)
- Explainability output (feature contributions, matched/missing keywords, evidence points)
- ATS compatibility scoring + keyword density analysis
- Optional Groq-powered personalized improvement suggestions
- Batch screening mode for ranking multiple resumes against one JD
- Privacy controls (PII redaction + no data retention mode)
- FastAPI backend for production integrations
- Evaluation script with labeled pairs and objective metrics
- Tests + CI + Docker deployment support

## Tech Stack
- **App/UI**: Streamlit
- **Backend API**: FastAPI + Uvicorn
- **PDF Parsing**: PyMuPDF (`fitz`)
- **OCR Fallback**: `pytesseract` + Pillow (optional)
- **Embedding Models**: `sentence-transformers` (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- **Optional Reranker**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Metrics/ML Utils**: scikit-learn, SciPy, pandas, numpy
- **LLM Suggestions**: Groq SDK (Llama 3.1 / Mixtral)
- **Testing**: pytest
- **Containerization**: Docker + docker-compose
- **CI**: GitHub Actions
- **Python**: 3.11+

## Project Structure
```text
/
├── app.py
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── utils/
│   ├── extractor.py
│   ├── matcher.py
│   ├── llm_suggestions.py
│   ├── observability.py
│   ├── privacy.py
│   └── skills_db.py
├── data/
│   └── eval_pairs.jsonl
├── scripts/
│   └── evaluate.py
├── tests/
│   ├── test_api.py
│   ├── test_extractor.py
│   ├── test_matcher.py
│   └── test_privacy.py
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

## Architecture
```mermaid
flowchart TD
    A[Resume PDF(s) + JD Text/PDF] --> B[Extraction Layer]
    B --> C[Text Cleaning + Section Detection]
    C --> D[PII Redaction Optional]
    D --> E[Hybrid Scoring Engine]
    E --> E1[Semantic Similarity - SentenceTransformer]
    E --> E2[Lexical Similarity - TF-IDF]
    E --> E3[Skill Alignment - Ontology Match]
    E --> E4[Cross-Encoder Reranker Optional]
    E --> F[Explainability]
    F --> F1[Feature Contributions]
    F --> F2[Matched/Missing Keywords]
    F --> F3[Evidence Points]
    D --> G[ATS Analyzer]
    G --> G1[Section Score]
    G --> G2[Contact Score]
    G --> G3[Keyword Score]
    G --> G4[Length/Bullet Score]
    F --> H[LLM Suggestions Optional]
    G --> I[Streamlit UI + Downloads]
    H --> I
    E --> J[FastAPI /analyze Endpoint]
    I --> K[Reports: Markdown / PDF / JSON / CSV]
```

## Features
1. Single resume analysis with full dashboard
2. Batch resume screening leaderboard
3. Match score (0-100) with component visibility
4. Skill overlap and missing skill detection
5. ATS compatibility and keyword density
6. Explainability tab with contribution breakdown
7. Personalized suggestions (Groq or fallback)
8. Privacy redaction mode
9. No-data-retention mode
10. Observability metrics (stage latency + LLM token/cost)

## Results and Evaluation
Use the included evaluation dataset and script:

```bash
python scripts/evaluate.py --dataset data/eval_pairs.jsonl --output artifacts/eval_results.csv
```

Output includes:
- MAE
- RMSE
- Pearson correlation
- Spearman correlation
- Pair-level predictions CSV for analysis

## Local Setup
### 1) Create virtual environment
```bash
python -m venv .venv
```

Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run Streamlit app
```bash
streamlit run app.py
```

### 4) Run FastAPI backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5) Run tests
```bash
pytest
```

## API Usage Example
### POST `/analyze`
```json
{
  "resume_text": "Python FastAPI engineer with NLP experience...",
  "jd_text": "Looking for AI engineer with Python, NLP, FastAPI...",
  "embedding_model": "all-MiniLM-L6-v2",
  "use_reranker": true,
  "redact_pii": true,
  "include_suggestions": false
}
```

Response includes:
- `score_details`
- `skill_details`
- `ats_details`
- `keyword_rows`
- `strong_points`
- `explanation`
- `suggestions_payload`
- `privacy`
- `observability`

## Docker
### Build and run app
```bash
docker build -t resume-screener .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key resume-screener
```

### Run both app + API
```bash
docker-compose up --build
```

## Streamlit Community Cloud Deployment (Free)
1. Push the repository to GitHub.
2. Open `https://share.streamlit.io`.
3. Create new app and set main file path to `app.py`.
4. Add optional secret:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```
5. Deploy.

## GitHub to Hugging Face Auto-Deploy (CI/CD)
This repo includes `.github/workflows/deploy-hf.yml` for automatic Space deployment on every push to `main`/`master`.

### Required GitHub Secrets
1. `HF_TOKEN`: Hugging Face write token
2. `HF_SPACE_REPO`: Hugging Face Space repo in `username/space-name` format

### Setup Steps
1. Create a Hugging Face Space (SDK: Docker).
2. In GitHub repo settings, add the two secrets above.
3. Push to `main` (or run workflow manually via `workflow_dispatch`).
4. Workflow runs tests, syncs files, and pushes to the Space repo.

### Hugging Face Space Metadata
- Space README metadata template is stored at `.hf/README.md`.
- During deploy, workflow copies it to `README.md` inside the Space.

## Screenshots
Add screenshots under `assets/screenshots/` and reference:
```md
![Input](assets/screenshots/01-input.png)
![Overview](assets/screenshots/02-overview.png)
![Explainability](assets/screenshots/03-explainability.png)
![Batch](assets/screenshots/04-batch.png)
```

## Future Improvements
1. Domain-specific calibration sets (e.g., Data, Backend, Product)
2. Fine-tuned reranker on labeled recruiter judgments
3. Bias and fairness diagnostics by candidate cohort
4. Persistent telemetry + dashboards (Prometheus/Grafana)
5. Auth + role-based recruiter workspace
6. Candidate history search with vector database

## Live Demo Link
Set after deployment:
`https://<your-app-name>.streamlit.app`
