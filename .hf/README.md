---
title: AI Resume Screener & Job Matcher
emoji: 📄
colorFrom: emerald
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# AI Resume Screener & Job Matcher

This Space is auto-deployed from GitHub via GitHub Actions.

## Features
- Resume-to-JD hybrid match scoring
- Skill gaps and ATS insights
- Explainability + confidence scoring
- Resume rewrite diff and tailored draft export
- Batch recruiter screening dashboard

## Notes
- Configure `GROQ_API_KEY` in Space Secrets if you want LLM suggestions.
- This app runs as a Docker Space and serves Streamlit on port `8501`.
