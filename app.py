"""Streamlit chat interface for the NADRA FAQ assistant.

Run with:
    streamlit run app.py
"""

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag_pipeline import answer_question, warm_up  # noqa: E402


st.set_page_config(
    page_title="NADRA Guide | AI Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

    :root {
        --green-950: #063b2a;
        --green-800: #086044;
        --green-650: #0b7a53;
        --green-100: #dff3e9;
        --green-50: #f2faf6;
        --ink: #16332a;
        --muted: #61756e;
        --line: #dbe9e2;
    }

    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 88% 4%, rgba(25, 147, 97, .12), transparent 25rem),
            linear-gradient(180deg, #f8fcfa 0%, #f3f8f5 100%);
        color: var(--ink);
    }

    .stApp p, .stApp li, .stApp label, .stApp span {
        color: var(--ink);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #073f2d 0%, #075239 58%, #086848 100%);
        border-right: 0;
    }

    [data-testid="stSidebar"] * {
        color: #f5fffa !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 3.2rem;
        padding: .7rem .85rem;
        background: rgba(255,255,255,.09);
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 12px;
        color: #f7fff9;
        text-align: left;
        justify-content: flex-start;
        font-weight: 500;
        transition: all .18s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,.16);
        border-color: rgba(255,255,255,.34);
        transform: translateY(-1px);
    }

    .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: .8rem;
        padding: .45rem .1rem 1.5rem;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(145deg, #29a76f, #0a754f);
        border: 1px solid rgba(255,255,255,.22);
        box-shadow: 0 10px 24px rgba(0,0,0,.15);
        font-size: 1.45rem;
    }

    .brand-name {
        font-family: "Manrope", sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .brand-subtitle {
        margin-top: .25rem;
        color: rgba(245,255,250,.68) !important;
        font-size: .78rem;
    }

    .side-label {
        margin: .6rem 0 .7rem;
        color: rgba(245,255,250,.64) !important;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .11em;
        text-transform: uppercase;
    }

    .privacy-note {
        margin-top: 1.2rem;
        padding: .9rem 1rem;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 12px;
        background: rgba(0,0,0,.08);
        color: rgba(245,255,250,.72) !important;
        font-size: .78rem;
        line-height: 1.5;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.65rem 1.85rem;
        border: 1px solid rgba(8, 96, 68, .10);
        border-radius: 24px;
        background: linear-gradient(130deg, #ffffff 0%, #f2fbf6 100%);
        box-shadow: 0 16px 50px rgba(23, 77, 56, .08);
        margin-bottom: 1rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -55px;
        top: -75px;
        border: 34px solid rgba(10, 122, 82, .07);
        border-radius: 50%;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .35rem .7rem;
        border-radius: 999px;
        background: var(--green-100);
        color: var(--green-800);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }

    .hero h1 {
        max-width: 650px;
        margin: .85rem 0 .55rem;
        font-family: "Manrope", sans-serif;
        color: var(--green-950);
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.08;
        letter-spacing: -.04em;
    }

    .hero p {
        max-width: 690px;
        margin: 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
    }

    .service-row {
        display: flex;
        flex-wrap: wrap;
        gap: .48rem;
        margin-top: 1.15rem;
    }

    .service-pill {
        padding: .35rem .68rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255,255,255,.75);
        color: var(--green-800);
        font-size: .75rem;
        font-weight: 700;
    }

    .welcome-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .8rem;
        margin: .9rem 0 1.2rem;
    }

    .info-card {
        padding: 1rem 1.05rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255,255,255,.70);
    }

    .info-card strong {
        display: block;
        margin-bottom: .25rem;
        color: var(--green-950);
        font-size: .88rem;
    }

    .info-card span {
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.4;
    }

    [data-testid="stChatMessage"] {
        margin: .7rem 0;
        padding: 1rem 1.2rem;
        border: 1px solid #cfe2d9;
        border-radius: 18px;
        background: #ffffff !important;
        color: var(--ink) !important;
        box-shadow: 0 8px 24px rgba(25, 76, 56, .09);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #173c30 !important;
        opacity: 1 !important;
        line-height: 1.65;
    }

    /* User messages are bold, high-contrast green bubbles. Multiple selectors
       keep this working across Streamlit versions. */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"][aria-label*="user" i] {
        margin-left: clamp(1rem, 8vw, 5rem);
        border-color: #08704d;
        background: linear-gradient(135deg, #086044, #0a7b53) !important;
        box-shadow: 0 10px 28px rgba(8, 96, 68, .20);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div,
    [data-testid="stChatMessage"][aria-label*="user" i] p,
    [data-testid="stChatMessage"][aria-label*="user" i] span,
    [data-testid="stChatMessage"][aria-label*="user" i] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessage"][aria-label*="assistant" i] {
        margin-right: clamp(1rem, 5vw, 3rem);
        border-left: 5px solid #19a56f;
        background: linear-gradient(135deg, #ffffff, #effaf4) !important;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background: rgba(255,255,255,.18) !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: #d9f4e6 !important;
    }

    [data-testid="stExpander"] {
        margin-top: .7rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--green-50);
    }

    [data-testid="stExpander"] * {
        color: #214c3d !important;
    }

    .source-item {
        margin: .35rem 0;
        padding: .55rem .7rem;
        border-radius: 9px;
        background: #fff;
        color: #456258;
        font-size: .82rem;
    }

    [data-testid="stChatInput"] {
        border: 2px solid #78ad94;
        border-radius: 18px;
        background: #ffffff !important;
        box-shadow: 0 14px 38px rgba(11, 82, 56, .20);
        overflow: hidden;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--green-650);
        box-shadow: 0 12px 35px rgba(11, 122, 83, .20);
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #153c2f !important;
        -webkit-text-fill-color: #153c2f !important;
        caret-color: #08704d !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #668078 !important;
        -webkit-text-fill-color: #668078 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background: #08704d !important;
        color: #ffffff !important;
        border-radius: 11px !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInput"] button svg * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background: linear-gradient(180deg, rgba(243,248,245,0), #f3f8f5 34%) !important;
    }

    [data-testid="stToolbar"], .stDeployButton {
        display: none !important;
    }

    .stSpinner > div {
        color: var(--green-650);
    }

    @media (max-width: 700px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 1.45rem; border-radius: 19px; }
        .welcome-grid { grid-template-columns: 1fr; }
        [data-testid="stChatMessage"] { margin-left: 0 !important; margin-right: 0 !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the multilingual knowledge base…")
def _warm_up_once():
    warm_up()
    return True


def show_sources(sources: list[dict]) -> None:
    """Render source references consistently and safely."""
    if not sources:
        return
    with st.expander(f"View sources ({len(sources)})"):
        for source in sources:
            filename = html.escape(str(source.get("source", "Official document")))
            page = html.escape(str(source.get("page", "—")))
            st.markdown(
                f'<div class="source-item">📄 {filename} &nbsp;·&nbsp; Page {page}</div>',
                unsafe_allow_html=True,
            )


_warm_up_once()

if "history" not in st.session_state:
    st.session_state.history = []

examples = [
    "What documents are required to renew a CNIC?",
    "NICOP banwane ki fees aur process kya hai?",
    "How can I track my NADRA application?",
    "FRC ke liye kaun eligible hai?",
]

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🇵🇰</div>
            <div>
                <div class="brand-name">NADRA Guide</div>
                <div class="brand-subtitle">AI information assistant</div>
            </div>
        </div>
        <div class="side-label">Try asking</div>
        """,
        unsafe_allow_html=True,
    )

    for index, example in enumerate(examples):
        if st.button(example, key=f"example_{index}"):
            st.session_state.queued_question = example

    st.markdown('<div class="side-label">Conversation</div>', unsafe_allow_html=True)
    if st.button("＋ Start a new chat", key="clear_chat"):
        st.session_state.history = []
        st.session_state.pop("queued_question", None)
        st.rerun()

    st.markdown(
        """
        <div class="privacy-note">
            🔒 <strong>Privacy reminder</strong><br>
            Do not share CNIC numbers, passwords, biometric data, or other sensitive personal information.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">● Document-grounded answers</div>
        <h1>Identity services, explained simply.</h1>
        <p>
            Ask about documents, eligibility, fees, timelines, and application
            procedures in English, اردو, or Roman Urdu.
        </p>
        <div class="service-row">
            <span class="service-pill">CNIC</span>
            <span class="service-pill">NICOP</span>
            <span class="service-pill">POC</span>
            <span class="service-pill">CRC</span>
            <span class="service-pill">FRC</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.history:
    st.markdown(
        """
        <div class="welcome-grid">
            <div class="info-card">
                <strong>🌐 Multilingual</strong>
                <span>Ask naturally in English, Urdu, or Roman Urdu.</span>
            </div>
            <div class="info-card">
                <strong>📚 Source-aware</strong>
                <span>See the document and page used for each answer.</span>
            </div>
            <div class="info-card">
                <strong>⚡ Quick guidance</strong>
                <span>Get concise help across common identity services.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.history:
    avatar = "👤" if message["role"] == "user" else "🇵🇰"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

typed_question = st.chat_input("Ask about a NADRA service…")
question = typed_question or st.session_state.pop("queued_question", None)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🇵🇰"):
        with st.spinner("Searching the official document collection…"):
            try:
                result = answer_question(question)
            except Exception as exc:  # Keep failures readable in the UI.
                error_message = (
                    "I couldn't complete that request. Please check the API connection "
                    "and try again."
                )
                st.error(error_message)
                st.session_state.history.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
            else:
                st.markdown(result["answer"])
                show_sources(result.get("sources", []))
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )
    st.rerun()
