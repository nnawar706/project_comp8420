"""
Customer support assistant, built as a live chat over src/pipeline.py.

The customer types a message in the chat box and the assistant answers in the
conversation. Each reply is produced by the full pipeline (classify, sentiment,
entity extraction, policy retrieval, grounded reply, and the auto-resolve or
escalate decision). A short strip under each reply shows what the system decided,
with the supporting detail available on demand.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.generation import ollama_available

st.set_page_config(page_title="Customer Support Assistant", layout="centered")

st.markdown("""
<style>
  .block-container {padding-top: 3.2rem; max-width: 820px;}
  .topbar {display:flex; align-items:center; gap:.6rem; padding:.8rem 1.1rem;
           background:linear-gradient(120deg,#1d3557,#2a6f97); color:#fff;
           border-radius:12px; margin:.4rem 0 1rem;}
  .topbar .name {font-weight:700; font-size:1.08rem;}
  .topbar .status {margin-left:auto; font-size:.82rem; opacity:.95;}
  .dot {height:9px; width:9px; border-radius:50%; display:inline-block; margin-right:5px;}
  .assist {display:flex; gap:.4rem; flex-wrap:wrap; margin:.2rem 0;}
  .chip {display:inline-block; padding:.12rem .55rem; border-radius:999px;
         font-size:.72rem; font-weight:600; color:#fff;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting the assistant...")
def get_pipeline(use_rag, variant, backend):
    from src.pipeline import CustomerServicePipeline
    return CustomerServicePipeline(use_rag=use_rag, prompt_variant=variant,
                                   sentiment_backend=backend)


def sent_color(label):
    return {"positive": "#2a9d8f", "negative": "#d62828", "neutral": "#8d99ae"}.get(label, "#8d99ae")


def render_assist(res):
    cat = f"<span class='chip' style='background:#2a6f97'>{res.category} {res.category_confidence:.0%}</span>"
    sen = (f"<span class='chip' style='background:{sent_color(res.sentiment['label'])}'>"
           f"{res.sentiment['label']}</span>")
    dec = ("<span class='chip' style='background:#d62828'>escalated to a human</span>"
           if res.should_escalate else
           "<span class='chip' style='background:#2a9d8f'>resolved automatically</span>")
    st.markdown(f"<div class='assist'>{cat}{sen}{dec}</div>", unsafe_allow_html=True)
    with st.expander("Details"):
        if res.entities:
            st.markdown("**Detected details**")
            st.dataframe([{"detail": e["text"], "type": e["label"]} for e in res.entities],
                         use_container_width=True, hide_index=True)
        if res.retrieved:
            st.markdown("**Policy referenced**")
            for c in res.retrieved:
                st.caption(f"{c['source']} (match {c['score']})")
        if res.should_escalate and res.escalation_reasons:
            st.markdown("**Why a human was looped in**")
            for r in res.escalation_reasons:
                st.caption(r)


online = ollama_available()

st.sidebar.title("Console")
st.sidebar.markdown(
    f"<span class='dot' style='background:{'#2a9d8f' if online else '#d62828'}'></span>"
    f"{'Assistant online' if online else 'Language model offline, replies go to a human'}",
    unsafe_allow_html=True)
st.sidebar.caption(f"Model: {config.LLM_MODEL}")
with st.sidebar.expander("Advanced settings"):
    use_rag = st.toggle("Ground replies in company policy (RAG)", value=True)
    variant = st.selectbox("Reasoning style", ["few_shot_cot", "few_shot", "zero_shot"])
    backend = st.selectbox("Sentiment engine", ["vader", "transformer"])
if st.sidebar.button("New conversation", use_container_width=True):
    st.session_state.history = []
    st.rerun()

st.markdown(f"""
<div class="topbar">
  <span class="name">Customer Support Assistant</span>
  <span class="status"><span class='dot' style='background:{'#7CFC9B' if online else '#ff9a9a'}'></span>
  {'Online' if online else 'Limited'}</span>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

WELCOME = ("Hi, I'm your support assistant. I can help with orders, deliveries, refunds, "
           "returns, payments and your account. What can I do for you today?")

if not st.session_state.history:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("res") is not None:
            render_assist(turn["res"])

prompt = st.chat_input("Type your message...")
if prompt and prompt.strip():
    st.session_state.history.append({"role": "user", "text": prompt, "res": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    pipe = get_pipeline(use_rag, variant, backend)
    with st.chat_message("assistant"):
        with st.spinner("Looking into that..."):
            res = pipe.process(prompt)
        reply = res.response or ("Let me connect you with a specialist who can help with "
                                 "this right away.")
        if res.should_escalate:
            reply = f"{reply}\n\n*I'm connecting you with a human specialist now.*"
        st.markdown(reply)
        render_assist(res)
    st.session_state.history.append({"role": "assistant", "text": reply, "res": res})
