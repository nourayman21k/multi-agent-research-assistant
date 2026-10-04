"""Streamlit frontend for the research assistant API.

Run (from the project1 folder, with the API already running):
    streamlit run app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

EXAMPLES = [
    "What is the transformer architecture, and which well-known AI models are built on it today?",
    "How does Self-RAG differ from Corrective RAG?",
    "What is ReAct and who proposed it?",
]

st.set_page_config(page_title="Research Assistant", page_icon="🔎")
st.title("🔎 Research Assistant")
st.caption("Planner → parallel Researchers → Synthesizer → Reflector · sources: 5 AI papers, Wikipedia, the web")

# ───────────── sidebar: backend status + examples ─────────────
with st.sidebar:
    st.subheader("Backend")
    try:
        requests.get(f"{API_URL}/health", timeout=3).raise_for_status()
        st.success(f"Connected to {API_URL}")
    except requests.RequestException:
        st.error(f"API not reachable at {API_URL}.\n\nStart it with `uvicorn api:app --port 8000`.")

    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state.question = ex

# ───────────── question form ─────────────
with st.form("ask"):
    question = st.text_area("Your question", key="question", height=100)
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    with st.spinner("Researching... this usually takes 1-3 minutes."):
        try:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=600)
            resp.raise_for_status()
            st.session_state.result = resp.json()
        except requests.HTTPError:
            st.error(f"API error: {resp.json().get('detail', resp.text)}")
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")
elif submitted:
    st.warning("Please type a question.")

# ───────────── result ─────────────
result = st.session_state.get("result")
if result:
    st.divider()
    st.markdown(result["answer"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Sub-questions", len(result["sub_questions"]))
    c2.metric("Evidence items", len(result["evidence"]))
    c3.metric("Reviews", result["reflections"], help=f"Reflector's last decision: {result['decision']}")

    with st.expander("🗺️ Plan: sub-questions"):
        for q in result["sub_questions"]:
            n = sum(e["sub_question"] == q for e in result["evidence"])
            st.markdown(f"- {q}  \n  <small>{n} evidence items</small>", unsafe_allow_html=True)

    with st.expander("📚 Evidence the researchers kept"):
        for q in result["sub_questions"]:
            items = [e for e in result["evidence"] if e["sub_question"] == q]
            if not items:
                continue
            st.markdown(f"**{q}**")
            for e in items:
                st.caption(e["tool"] or "tool")
                st.text(e["text"])
