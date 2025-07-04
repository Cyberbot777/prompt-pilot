# PromptPilot Production Notes

**Project Status:** Updated Juky 4, 2025

---

## Overview

PromptPilot is a developer-facing tool for testing and refining prompt + system instruction configurations for LLMs (starting with OpenAI’s GPT-3.5).  
It offers a real-time chat interface alongside editable instruction fields, structured in a two-pane layout.

This document details:
- System architecture
- UI behavior
- Custom styling decisions
- Backend API integration
- File import/export logic for prompt sessions
- Production deployment + custom domain setup
- Multi-model Claude integration with model labeling

---

## 🛠️ Tech Stack

- **Frontend:** React (Vite) + TypeScript + TailwindCSS + Framer Motion
- **Backend:** FastAPI (Python 3.11), served via Uvicorn
- **LLM API:** OpenAI `gpt-3.5-turbo`, Anthropic `claude-3-5-sonnet-20241022`
- **Deployment Target:** Vercel (frontend), Render (backend)
- **Styling:** TailwindCSS + custom scrollbars, dark terminal UI

---

## API Integration

**POST /api/generate**

- Accepts:
  ```json
  {
    "systemInstruction": "string",
    "messages": [{ "role": "user" | "assistant", "content": "string" }],
    "model": "gpt-3.5-turbo" | "claude-3-5-sonnet-20241022"
  }
  ```
- Routes to:
  - OpenAI Chat API for GPT-3.5
  - Anthropic Messages API for Claude
- Returns:
  ```json
  {
    "message": "string"
  }
  ```

**Security:**
- API keys loaded via `.env`:
  - `OPENAI_API_KEY`
  - `CLAUDE_API_KEY`
- Backend proxy isolates secret keys from frontend
- CORS configured for:
  - `http://localhost:5173`
  - `https://prompt-pilot-pi.vercel.app`
  - `https://prompt-pilot.ai`
  - `https://www.prompt-pilot.ai`

---

## UI Layout

### ➤ Left Panel (Prompt Editor)

- System instruction input
  - Monospaced font
  - Terminal style header: `user@prompt-lab:~$ System Instruction:`
  - Auto-expands with `scrollHeight` logic
  - Custom green scrollbar (`.custom-scroll`)
  - Max-height: 70vh with vertical scroll fallback
  - **Supports import from `.txt`, `.json`, or `.md`**
  - **Auto-expands after import**

### ➤ Right Panel (AI Chat)

- Model toggle buttons above output
- User input field
  - Expandable with shift+enter
  - Styled scroll ring
- AI output area
  - Animated GPT/Claude responses
  - Model name labeling
  - **Auto-scrolls to latest message**

### ➤ Toolbar (Top Right)

- **Save Icon**:  
  Triggers dropdown with export options:
  - `.json` — exports full session (instruction, model, messages)
  - `.txt` — exports only the system instruction

- **Upload Icon**:  
  Opens file picker to import `.txt`, `.json`, or `.md`

- **Export menu**:  
  Closes on outside click or Escape

---

## UX Behavior

- Model selection updates placeholder dynamically
- Claude and GPT responses labeled clearly
- Replies animate with `setInterval(40ms)`
- Instruction editing does not clear session
- Auto-scroll enabled while response is typing
- Export dropdown uses keyboard-style dropdown UX (no popups)
- Full `.json` sessions can be re-imported (systemInstruction only)
- `.txt` export provides writer-friendly copy of instructions

---

## Deployment Notes

- **Frontend:** Vercel  
  - Connected to GitHub `main` branch  
  - `VITE_API_URL` points to backend
- **Backend:** Render  
  - GitHub-deployed with `.env` for keys
  - CORS supports localhost + production

---

## Custom Domain

- Registered: `prompt-pilot.ai`
- Connected via Vercel:
  - `A Record`: `216.198.79.193`
  - `CNAME`: `vercel-dns`
- Live at:
  https://prompt-pilot.ai  
  https://www.prompt-pilot.ai

---

## Upcoming Feature: Prompt Tudor (v2.5+)

Prompt Tudor is an experimental new feature planned for PromptPilot that allows users to analyze and optimize their prompts before submitting them to GPT or Claude models.

Powered by the **Iris Agent (Phase 1)** — originally developed in the Prompt Fractal project — Prompt Tudor will:
- Review prompts for clarity, length, and ambiguity.
- Provide a professional analysis with a clarity rating and specific improvement suggestions.
- Automatically generate a fully rewritten, more effective version of the prompt.

This feature will help users refine unclear or overly complex prompts to improve model outputs, ensuring LLM responses are more relevant, respectful, and actionable.

**Status:**  
Prompt Tudor backend logic is already integrated and available for future UI wiring and testing.
