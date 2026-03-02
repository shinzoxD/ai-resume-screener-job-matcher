---
title: ResumePilot AI
emoji: "📄"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# ResumePilot AI

Dual-mode resume intelligence app:
- JD Match mode: resume vs job-description scoring
- Resume Health mode: standalone resume quality scoring and fixes

This Space is auto-deployed from GitHub Actions.

## Configure Secrets
- Add `GROQ_API_KEY` in Space Secrets for LLM suggestions.
- If missing, the app automatically uses rule-based suggestions.

## Source
- GitHub: https://github.com/shinzoxD/ai-resume-screener-job-matcher
