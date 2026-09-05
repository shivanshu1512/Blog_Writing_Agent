# ✍️ Blog Writing Agent

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blogwritingagent1.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An AI-powered multi-agent system that generates research-backed, long-form technical blog posts — fully automated, end to end.**

[🚀 Live Demo](https://blogwritingagent1.streamlit.app/) · [📂 GitHub](https://github.com/shivanshu1512/Blog_Writing_Agent)

</div>

---

## 📌 What is this?

Blog Writing Agent is a LangGraph-based multi-agent pipeline that takes a topic and produces a complete, publication-ready blog post with:

- 🔎 **Web research** (DuckDuckGo, no API key needed)
- 🧩 **Structured planning** (audience, tone, sections)
- ✍️ **Parallel section writing** (multiple AI workers)
- 🖼️ **AI-generated images** (Google Gemini)
- 📝 **Final markdown output** (download ready)

Built as a learning project to explore **agentic AI workflows**, **LangGraph state machines**, and **Streamlit deployment**.

---

## 🏗️ Architecture

```
Topic Input
    │
    ▼
┌─────────┐     needs research?
│ Router  │ ──────────────────► Web Research (DuckDuckGo)
└─────────┘                            │
    │ ◄─────────────────────────────────┘
    ▼
┌──────────────┐
│ Orchestrator │  →  Creates structured blog plan (title, tasks, audience, tone)
└──────────────┘
    │
    ▼  (parallel fanout)
┌──────────────────────────────┐
│ Worker  Worker  Worker  ...  │  →  Each writes one section in Markdown
└──────────────────────────────┘
    │
    ▼
┌────────────────────┐
│ Reducer + Imager   │  →  Merges sections, adds AI-generated images
└────────────────────┘
    │
    ▼
 Final Blog Post (.md)
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **LLM** | NVIDIA NIM (`openai/gpt-oss-20b`) — free tier |
| **Web Search** | DuckDuckGo (`ddgs`) — completely free, no key |
| **Image Generation** | Google Gemini 2.0 Flash — free tier |
| **Frontend** | [Streamlit](https://streamlit.io) |
| **Language** | Python 3.10+ |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/shivanshu1512/Blog_Writing_Agent.git
cd Blog_Writing_Agent
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root:
```env
NVIDIA_API_KEY="your-nvidia-nim-api-key"
GOOGLE_API_KEY="your-google-ai-studio-key"
```

> Get free keys:
> - NVIDIA NIM → https://build.nvidia.com/settings/api-keys
> - Google AI Studio → https://aistudio.google.com/app/apikey

**5. Run the app**
```bash
streamlit run app.py
```

Open → **http://localhost:8501** 🎉

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your fork, branch `main`, file `app.py`
4. Add secrets in **Settings → Secrets**:
```toml
NVIDIA_API_KEY = "your-key-here"
GOOGLE_API_KEY = "your-key-here"
```
5. Click **Deploy** ✅

---

## 📸 Features

| Feature | Description |
|---|---|
| 🧩 Plan Tab | View the AI-generated blog plan — audience, tone, tasks |
| 🔎 Evidence Tab | See all web sources gathered during research |
| 📝 Preview Tab | Read the full rendered blog with images |
| 🖼️ Images Tab | View & download AI-generated diagrams |
| 🧾 Logs Tab | Live event log from the agent graph |
| ⬇️ Download | Export as `.md` or full bundle (MD + images) |
| 📂 Past Blogs | Load and re-view previously generated blogs |

---

## 📁 Project Structure

```
Blog_Writing_Agent/
├── app.py              # Streamlit frontend (clean UI)
├── bwa_backend.py      # LangGraph agent pipeline
├── bwa_frontend.py     # Original frontend (reference)
├── requirements.txt    # Python dependencies
├── .gitignore
└── *.ipynb             # Experiment notebooks
```

---

## 🧑‍💻 About

Built by **Shivanshu Shukla** — exploring AI agents, LangGraph, and production-ready deployments.

[![GitHub](https://img.shields.io/badge/GitHub-shivanshu1512-black?logo=github)](https://github.com/shivanshu1512)

---

<div align="center">
  <sub>⭐ If you found this useful, give it a star!</sub>
</div>
