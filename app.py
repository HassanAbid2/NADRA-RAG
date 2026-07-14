"""Streamlit demo chat UI for the NADRA FAQ system.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag_pipeline import answer_question, warm_up  # noqa: E402

st.set_page_config(page_title="NADRA FAQ Assistant", page_icon="🇵🇰")


@st.cache_resource(show_spinner="Loading knowledge base (one-time) ...")
def _warm_up_once():
    warm_up()
    return True


_warm_up_once()
st.title("🇵🇰 NADRA FAQ Assistant")
st.caption(
    "Ask about CNIC, NICOP, POC, CRC, and FRC — documents, fees, eligibility, "
    "timelines, and procedures. Answers come only from official NADRA documents."
)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']} (page {s['page']})")

if question := st.chat_input("e.g. What documents do I need to renew my CNIC?"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching NADRA documents..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.write(f"- {s['source']} (page {s['page']})")
    st.session_state.history.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
