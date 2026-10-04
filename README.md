# Multi-agent research assistant

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/nourayman21k/multi-agent-research-assistant)

Ask a question and a small team of AI agents answers it with citations. They search 5 AI research papers, Wikipedia and the web, then review their own answer before showing it to you.

Built with **LangGraph**, served by a **FastAPI** backend, with a **Streamlit** UI.

## How it works

```
Planner ──► Researcher (sub-question 1) ──┐
        ├─► Researcher (sub-question 2) ──┼──► Synthesizer ──► Reflector ──► answer
        └─► Researcher (sub-question 3) ──┘         ▲              │
                     ▲                              └── revise ────┤
                     └──────── research more (follow-up questions) ┘
```

| Agent | What it does |
|---|---|
| **Planner** | Splits the question into 1-4 self-contained sub-questions. |
| **Researcher** | One per sub-question, all running **in parallel**. Each picks tools (paper search, Wikipedia, DuckDuckGo), and a **Grader** keeps only the relevant results. |
| **Synthesizer** | Writes the answer from all the evidence, with `[n]` citations. The references list is built in code, so sources can't be invented. |
| **Reflector** | Reviews the answer. It can **accept** it, ask for a **revision** (unsupported claims), or send **follow-up questions** to new researchers. |

**Models:** Qwen3-32B (Hugging Face) for the researchers and gpt-oss-120b (Groq) for the other agents.
**Papers:** Attention Is All You Need, RAG, ReAct, Self-RAG, Corrective RAG (in [`project1/`](project1)).

## You need two free API keys

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Create a fine-grained token with the **"Make calls to Inference Providers"** permission. |

## Option 1: try it in your browser (GitHub Codespaces, nothing to install)

1. Click **Open in GitHub Codespaces** above, or go to **Code → Codespaces → Create codespace on main**.
2. GitHub asks for `GROQ_API_KEY` and `HF_TOKEN`. Paste your keys and create the codespace.
   The first start installs everything, which takes about 5-10 minutes.
3. In the terminal, start the backend:
   ```bash
   cd project1
   uv run uvicorn api:app --port 8000
   ```
   The first start indexes the papers (about 1-2 minutes). Wait for `Application startup complete`.
4. Open a **second terminal** (the **+** button) and start the UI:
   ```bash
   cd project1
   uv run streamlit run app.py
   ```
   Codespaces then offers **Open in Browser** for port 8501. You can also find it in the **Ports** tab.

> Codespaces is free for personal accounts up to a monthly limit. Stop or delete the codespace when you're done.

## Option 2: run it on your own computer

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.14 for you.

```bash
git clone https://github.com/nourayman21k/multi-agent-research-assistant.git
cd multi-agent-research-assistant
uv sync
```

Copy `.env.example` to `.env` and paste in your two keys. Then, in two terminals:

```bash
# terminal 1: backend
cd project1
uv run uvicorn api:app --port 8000

# terminal 2: UI, then open http://localhost:8501
cd project1
uv run streamlit run app.py
```

## Using the API directly

With the backend running, interactive docs are at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How does Self-RAG differ from Corrective RAG?"}'
```

The response contains `answer` (markdown with citations and references), `sub_questions`, `evidence`, `reflections` and `decision`.

## Project layout

```
project1/
  research_graph.py   the multi-agent LangGraph graph (models, tools, agents)
  api.py              FastAPI backend: GET /health, POST /ask
  app.py              Streamlit UI
  *.pdf               the 5 papers used for search
project1_again.ipynb  the same graph built step by step in a notebook
```

## Notes

- One question takes about 1-3 minutes and makes roughly 15-30 LLM calls. Free tiers can rate-limit you; if that happens, wait a minute and try again.
- The vector store is saved to `project1/chroma_db/` on the first run, so later starts are faster.
