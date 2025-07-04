# PromptPilot

🌐 **Live Site:** [https://prompt-pilot.ai](https://prompt-pilot.ai)

---

## Overview

**PromptPilot** is a full-stack AI customization tool that lets users design, test, and refine GPT agents using custom prompts and system instructions — all in a polished, side-by-side interface. Built with React, FastAPI, and PostgreSQL, the platform supports OpenAI and Claude API integration, secure prompt sessions, multi-format file import/export, and a user-friendly CLI-inspired UX.

PromptPilot is both a productivity tool and an educational experience: it empowers anyone — devs, writers, educators — to explore AI behavior design without needing to code.

---

## Features

- Prompt & System Inputs: Side-by-side layout with GPT and Claude (Grok planned)
- Model Toggle: Seamless GPT-3.5 and Claude 3.5 selection with persistent context
- Dynamic Labeling: Assistant responses labeled as GPT or Claude in real-time
- Markdown Output Viewer: Renders LLM responses cleanly
- Save Sessions: Export `.json` session files (with model, instruction, and messages)
- Upload `.txt`, `.json`, or `.md` files directly into the instruction panel
- Auto-expanding system instruction box with scrollHeight syncing
- CLI-style Save Dropdown: Choose `.json` or `.txt` export format from top-right
- Auth (v2): Planned JWT login system with user-specific session history and analytics
- Polished UI: Dark terminal-themed layout, smooth animations, keyboard-style dropdowns
- Deployment: Vercel (frontend), Render (backend), PostgreSQL DB

---

## Project Phases

- v1: Core GPT prompt trainer (local export/import)
- v1.5: Add Claude model support, toggle UI
- v2: JWT auth, user-specific session history (DB-backed)
- v2.5+: Model comparison, analytics, agent deploy

---

## Technologies

### Frontend
- React (Vite)
- Tailwind CSS (Dark + Glass UI)
- React Markdown (output rendering)
- Framer Motion (animations)
- Deployed via Vercel

### Backend
- FastAPI (Python)
- OpenAI + Anthropic Claude APIs (via backend proxy)
- PostgreSQL (session history)
- SQLAlchemy (ORM)
- Deployed via Render

---

## Skills Demonstrated

- Full-stack app architecture
- Multi-model API routing (GPT + Claude)
- Secure API proxying (OpenAI + Claude)
- Prompt engineering + structured testing
- Dynamic LLM labeling and UI feedback
- Dark-themed responsive UI
- RESTful API design with FastAPI + PostgreSQL
- JWT auth and session storage (v2)
- File export/import: `.json`, `.txt`, `.md`
- Real-world deployment using GitHub CI, Vercel, and Render

---

## Usage Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Cyberbot777/prompt-pilot.git
cd prompt-pilot
```

### 2. Set Up the Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit: [http://localhost:5173](http://localhost:5173)

### 3. Set Up the Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/Scripts/activate   # Use this in Git Bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend will run at: [http://localhost:8000](http://localhost:8000)

### 4. Environment Variables

Set these in `.env` files (frontend & backend as needed):

- `OPENAI_API_KEY`
- `CLAUDE_API_KEY`
- `DATABASE_URL`

---

## Upcoming Feature: Prompt Tudor (v2.5+)

Prompt Tudor is an experimental new feature planned for PromptPilot that allows users to analyze and optimize their prompts before submitting them to GPT or Claude models.

Powered by the **Iris Agent (Phase 1)** — originally developed in the Recursive project — Prompt Tudor will:
- Review prompts for clarity, length, and ambiguity.
- Provide a professional analysis with a clarity rating and specific improvement suggestions.
- Automatically generate a fully rewritten, more effective version of the prompt.

This feature will help users refine unclear or overly complex prompts to improve model outputs, ensuring LLM responses are more relevant, respectful, and actionable.

**Status:**  
Prompt Tudor backend logic is already integrated and available for future UI wiring and testing.


## Author

- Richard Hall

## Timeline

- Created: June 22, 2025  

---

> PromptPilot is more than a demo — it's a real tool for shaping AI with intention.