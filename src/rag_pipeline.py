"""NADRA RAG pipeline: multilingual retrieval and grounded answers via Groq.

Requires GROQ_API_KEY in .env (free key from https://console.groq.com).
"""

import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import kb

load_dotenv(kb.PROJECT_ROOT / ".env")

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = """You are an official FAQ assistant for NADRA (National Database and \
Registration Authority, Pakistan). You answer citizen questions about NADRA services: \
CNIC, NICOP, POC, CRC (Child Registration Certificate), and FRC (Family Registration \
Certificate) — covering required documents, eligibility, fees, processing timelines, \
and procedures.

STRICT RULES:

GROUNDING
1. Answer ONLY from the CONTEXT below — never outside knowledge. State fees, timelines, \
and document lists exactly as written; never round or change numbers.
2. Only when the context has NO relevant information, reply exactly (in the user's \
language): "I don't have verified NADRA information to answer that. Please contact the \
NADRA helpline (1777) or visit www.nadra.gov.pk." If the context covers the question \
only partially, answer the covered part and name what's missing — do NOT use that \
refusal sentence in that case.
2b. SCOPE: NADRA issues only CNIC, NICOP, POC, CRC, FRC and related registration \
records. Passports, driving licences, visas, tax filing, utility bills and other \
agencies' services are NOT NADRA services — refuse them with the rule-2 sentence even \
when the context looks superficially related. Renewing an identity card is NOT renewing \
a passport; never answer one as if it were the other. Never provide, guess, or look up \
any individual's personal data such as a CNIC number.

ACCURACY
3. Never invent or generalize a document name. When the context lists requirements that \
differ by scenario (resident vs non-resident, with vs without a blood relative), give \
EVERY scenario as its own labeled list with its exact documents — never call them \
unspecified when the context lists them.
4. Distinguish APPLYING from RECEIVING: content about collecting a finished card (token \
slip, authority letter, home delivery, "receiving of identity document") is NOT the list \
of documents needed to apply — do not present it as such.
5. If the context says an application auto-submits from records NADRA already holds \
(e.g. an FRC built from existing family data), lead with that — no separate checklist is \
needed beyond what's on file; only list what the user must supply for gaps the context \
names (such as adding an unregistered family member).

CLARITY & FORMAT
6. Write for an ordinary citizen. Lead with the direct answer in one plain sentence, then \
detail. Synthesize in your own words — do not paste raw fragments, table labels, or \
screen/section headings that don't answer the question. Say "not specified" once, not per \
item. Answer only what was asked; don't append unrequested sections. Use short numbered \
steps for procedures (give the app entry point first) and short bullets for lists. Use \
CONVERSATION HISTORY for follow-ups; if the user asks to repeat, translate, shorten, or \
restyle the previous answer, act on that answer, not a new topic.

LANGUAGE
7. Reply in the user's language: English → plain English; Urdu script → Pakistani Urdu in \
Urdu script; Roman-Urdu or mixed → natural Pakistani Roman Urdu (Latin letters), keeping \
English service names. If unsure, mirror the user. NEVER use Hindi or Devanagari.

SOURCES
8. End with a "Sources:" line listing the document names and pages used. For a \
language-only rewrite, keep the previous answer's citations.

LANGUAGE REQUIREMENT FOR THIS QUESTION:
{language_instruction}

INTERPRETED CURRENT QUESTION:
{interpreted_question}

CONVERSATION HISTORY:
{conversation_history}

CONTEXT:
{context}"""

_llm = None

_URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_HINDI_LEANING_ROMAN_TERMS = {
    "aavashyak",
    "avashyak",
    "dwara",
    "dusre",
    "doosre",
    "iske",
    "jaankari",
    "jankari",
    "kripya",
    "panjeekaran",
    "pramaan",
    "sabhi",
    "sabse",
    "sadasya",
    "shulk",
    "anya",
    "upalabdh",
    "uplabdh",
}
_ROMAN_URDU_REPLACEMENTS = {
    "aavashyak": "zaroori",
    "anya": "deegar",
    "avashyak": "zaroori",
    "doosre": "deegar",
    "dusre": "deegar",
    "dwara": "ke zariye",
    "iske": "is ke",
    "jaankari": "maloomat",
    "jankari": "maloomat",
    "kripya": "barah-e-karam",
    "panjeekaran": "registration",
    "pramaan": "saboot",
    "sabhi": "tamam",
    "sabse": "sab se",
    "sadasya": "member",
    "shulk": "fees",
    "upalabdh": "dastiyab",
    "uplabdh": "dastiyab",
}
_ROMAN_URDU_MARKERS = {
    "aap",
    "ap",
    "apna",
    "apni",
    "bana",
    "banwana",
    "banwane",
    "bata",
    "batao",
    "chahiye",
    "hai",
    "hain",
    "hoga",
    "hogay",
    "ka",
    "kaise",
    "kaun",
    "ke",
    "kesay",
    "kese",
    "ki",
    "kon",
    "kya",
    "liye",
    "main",
    "mein",
    "mera",
    "meri",
    "mujhe",
    "nahi",
    "par",
    "process",
    "se",
    "karun",
    "karoon",
    "karna",
    "karni",
    "kaha",
    "wala",
    "wali",
    "sakta",
    "sakte",
    "hoti",
    "hota",
}
# Unambiguous Roman-Urdu-only tokens: a single one is enough to classify a
# short question as Roman Urdu (they are never ordinary English words).
_STRONG_ROMAN_URDU = {
    "banaun", "banaon", "banana", "banau", "banana", "banwana", "banwane",
    "kaise", "kaisay", "kese", "kesay", "kaisa", "kahan", "kahaan", "kyun", "kyu",
    "chahiye", "chahiyay", "batao", "batau", "batayen", "mujhe", "mjhe",
    "karun", "karoon", "zaroori", "dastavez", "darkhwast", "maloomat",
}
_LANGUAGE_FOLLOWUP_RE = re.compile(
    r"\b(?:reply|answer|respond|translate|write|likho|batao)\b.*"
    r"\b(?:urdu|roman\s+urdu|english)\b|"
    r"\b(?:urdu|roman\s+urdu|english)\b.*\b(?:reply|answer|mein|main|me)\b|"
    # A short standalone language request ("in urdu", "urdu", "english mein").
    r"^\s*(?:in\s+|reply\s+in\s+|answer\s+in\s+)?(?:roman\s+urdu|urdu|english)"
    r"\s*(?:mein|main|me)?\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^\s*(?:a+o+a+|aoa|ass?alam(?:(?:u|\s*[ou])?\s*alaikum?)?|sal[aa]+m(?:\s*u?\s*alaikum?)?|"
    r"السلام\s+علیکم|سلام|hello+|hi+|hey+|yo+|sup|howdy|"
    r"good\s+(?:morning|afternoon|evening)|salaam)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_REFUSAL_MARKERS = (
    "i don't have verified nadra information",
    "verified nadra information is unavailable",
    "تصدیق شدہ",
    "مصدقہ معلومات",
)


def classify_question_language(question: str) -> str:
    """Classify the response style needed for English, Urdu, or Roman Urdu."""
    if _URDU_SCRIPT_RE.search(question):
        return "urdu"

    lowered = question.lower()
    if re.search(r"\broman[\s-]*urdu\b", lowered):
        return "roman_urdu"
    # Bare or prefixed language requests: "in urdu", "urdu mein", "reply in urdu".
    if re.search(r"\b(?:in\s+)?urdu\b(?:\s+(?:mein|main|me))?", lowered) and not (
        tokens := set(re.findall(r"[a-z]+", lowered))
    ) & (_ROMAN_URDU_MARKERS | _STRONG_ROMAN_URDU):
        return "urdu"
    if re.search(r"\b(?:reply|answer|respond|write)\s+in\s+urdu\b", lowered):
        return "urdu"
    if re.search(r"\burdu\s+(?:mein|main|me)\b", lowered):
        return "urdu"

    tokens = set(re.findall(r"[a-z]+", lowered))
    marker_count = len(tokens & _ROMAN_URDU_MARKERS)
    if marker_count >= 2 or tokens & _STRONG_ROMAN_URDU:
        return "roman_urdu"
    return "english"


def normalize_question(question: str) -> str:
    """Repair common service-name and Roman Urdu typos before retrieval."""
    normalized = re.sub(r"\bcn+n?ic\b", "CNIC", question, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\bkes[ae]y\b|\bkese\b",
        "kaise",
        normalized,
        flags=re.IGNORECASE,
    )
    # Roman Urdu modifiers the documents only ever state in English. Without this
    # "naya CNIC" misses the "New CNIC" fee row that "new CNIC" matches directly.
    normalized = re.sub(
        r"\bna(?:ya|ye|yi|i)\b", "new naya", normalized, flags=re.IGNORECASE
    )
    normalized = re.sub(
        r"\bpuran[ai]\b", "old purana", normalized, flags=re.IGNORECASE
    )
    # Informal city names used in queries but not in the source documents.
    normalized = re.sub(r"\bpindi\b", "Rawalpindi", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bbahawapur\b", "Bahawalpur", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bisb\b", "Islamabad", normalized, flags=re.IGNORECASE)
    return normalized


def is_language_followup(question: str) -> bool:
    """Return True for requests that only change the previous answer's language."""
    return bool(_LANGUAGE_FOLLOWUP_RE.search(question))


def is_greeting(question: str) -> bool:
    return bool(_GREETING_RE.match(question))


def contains_devanagari(text: str) -> bool:
    """Return True when a response contains Hindi/Devanagari characters."""
    return bool(_DEVANAGARI_RE.search(text))


def contains_urdu_script(text: str) -> bool:
    """Return True when the response contains Urdu/Arabic-script characters."""
    return bool(_URDU_SCRIPT_RE.search(text))


def contains_hindi_leaning_roman_terms(text: str) -> bool:
    """Detect common Hindi/Hinglish terms that should be Pakistani Roman Urdu."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & _HINDI_LEANING_ROMAN_TERMS)


def clean_roman_urdu_vocabulary(text: str) -> str:
    """Deterministically replace known Hindi/Hinglish terms after generation."""
    for hindi_term, urdu_term in _ROMAN_URDU_REPLACEMENTS.items():
        text = re.sub(
            rf"\b{re.escape(hindi_term)}\b",
            urdu_term,
            text,
            flags=re.IGNORECASE,
        )
    return text


def language_instruction(question: str) -> str:
    """Return a strict, question-specific response-language instruction."""
    language = classify_question_language(question)
    if language == "urdu":
        return (
            "Reply only in natural Pakistani Urdu using Urdu/Arabic script. "
            "Do not use Hindi wording or any Devanagari characters. Prefer Pakistani "
            "Urdu vocabulary. English NADRA service names such as CNIC and NICOP may "
            "remain in English."
        )
    if language == "roman_urdu":
        return (
            "Reply only in natural Pakistani Roman Urdu using Latin letters. "
            "Do not reply in Hindi, Hinglish, Devanagari, or Urdu script. Use "
            "Pakistani wording and keep familiar NADRA service names in English. "
            "Use terms such as tamam, maloomat, zaroori, darkhwast, dastavez, saboot, "
            "and fees. Avoid Hindi-leaning terms such as sabhi, jaankari, kripya, "
            "avashyak, pramaan, sadasya, panjeekaran, and shulk."
        )
    return "Reply in clear, plain English. Do not use Hindi or Devanagari."


def greeting_response(question: str) -> str:
    """Return a friendly greeting without attaching unrelated RAG sources."""
    language = classify_question_language(question)
    if language == "urdu":
        return "وعلیکم السلام! نادرا خدمات کے بارے میں میں آپ کی کیا مدد کر سکتا ہوں؟"
    if language == "roman_urdu" or re.search(
        r"a+o+a+|ass?alam|sal[aa]+m", question.lower()
    ):
        return (
            "Wa Alaikum Assalam! NADRA services ke hawale se main aap ki "
            "kya madad kar sakta hoon?"
        )
    return "Hello! How can I help you with NADRA services today?"


def format_conversation_history(history: list[dict] | None) -> str:
    """Format a bounded conversation transcript for contextual follow-ups."""
    if not history:
        return "(No previous conversation.)"

    lines = []
    for message in history[-12:]:
        role = str(message.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) or "(No previous conversation.)"


def previous_assistant_message(history: list[dict] | None) -> dict | None:
    if not history:
        return None
    for message in reversed(history):
        if message.get("role") == "assistant" and message.get("content"):
            return message
    return None


def sources_used_by_answer(docs, answer: str) -> list[dict]:
    """Return unique citations actually named by the generated answer."""
    answer_lower = answer.lower()
    if any(marker in answer_lower for marker in _REFUSAL_MARKERS):
        return []

    sources = []
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get("source")
        if not source or source in seen_sources or str(source).lower() not in answer_lower:
            continue
        seen_sources.add(source)

        source_lines = [
            line
            for line in answer.splitlines()
            if str(source).lower() in line.lower()
        ]
        cited_pages: list[int] = []
        for line in source_lines:
            tail = line.lower().split(str(source).lower(), 1)[1]
            match = re.search(
                r"(?:pages?|صفح(?:ہ|ات))\s*[:\-]?\s*([0-9][0-9,\s–—-]*)",
                tail,
            )
            if not match:
                continue
            for part in re.split(r"\s*,\s*", match.group(1).strip()):
                range_match = re.fullmatch(r"(\d+)\s*[–—-]\s*(\d+)", part)
                if range_match:
                    start, end = map(int, range_match.groups())
                    cited_pages.extend(range(start, end + 1))
                elif part.isdigit():
                    cited_pages.append(int(part))

        if cited_pages:
            available_pages = {
                candidate.metadata.get("page")
                for candidate in docs
                if candidate.metadata.get("source") == source
            }
            # One entry per document, pages joined — not one card per page.
            pages = [p for p in dict.fromkeys(cited_pages) if p in available_pages]
            if pages:
                sources.append({
                    "source": source,
                    "page": ", ".join(str(p) for p in pages),
                })
                continue
        sources.append({"source": source, "page": None})
    return sources


def strip_sources_line(answer: str) -> str:
    """Drop the trailing Sources: block — the frontend renders sources separately.

    Handles the English label plus the Urdu/Roman-Urdu equivalents the model
    produces when answering in those languages.
    """
    return re.split(
        r"\n+\s*(?:sources?|مصادر|ماخذ|ذرائع|حوالہ(?:\s*جات)?|maloomat\s+ka\s+zariya)\s*[:：]?",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, temperature=0)
    return _llm


def warm_up():
    """Load the embedding model, vector store, and BM25 index up front."""
    kb.get_retriever().search("warm up", k=1)


def retrieve(question: str, k: int = TOP_K):
    """Return the top-k multilingual hybrid-retrieval chunks."""
    return kb.get_retriever().search(question, k=k)


def format_context(docs) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{src}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    question: str,
    k: int = TOP_K,
    history: list[dict] | None = None,
) -> dict:
    """Answer with recent conversation context and grounded NADRA sources."""
    if is_greeting(question):
        return {"answer": greeting_response(question), "sources": []}

    interpreted_question = normalize_question(question)
    previous_answer = previous_assistant_message(history)
    language_followup = is_language_followup(question) and previous_answer is not None

    if language_followup:
        docs = []
        context = (
            "This is a language-only follow-up. Rewrite the previous assistant "
            "answer from CONVERSATION HISTORY without adding or removing facts."
        )
    else:
        docs = retrieve(interpreted_question, k=k)
        context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = prompt | _get_llm()
    inputs = {
        "context": context,
        "question": question,
        "language_instruction": language_instruction(question),
        "interpreted_question": interpreted_question,
        "conversation_history": format_conversation_history(history),
    }
    response = chain.invoke(inputs)

    target_language = classify_question_language(question)
    needs_language_rewrite = (
        contains_devanagari(response.content)
        or (target_language == "urdu" and not contains_urdu_script(response.content))
        or (target_language == "roman_urdu" and contains_urdu_script(response.content))
        or (
            target_language == "roman_urdu"
            and contains_hindi_leaning_roman_terms(response.content)
        )
    )
    if target_language in {"urdu", "roman_urdu"} and needs_language_rewrite:
        correction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
                ("assistant", "{draft_answer}"),
                (
                    "human",
                    "Rewrite that answer to follow the LANGUAGE REQUIREMENT exactly. "
                    "Remove every Hindi/Devanagari character and replace Hindi-leaning "
                    "vocabulary with natural Pakistani Urdu or Roman Urdu. If the "
                    "target is Urdu, write the complete answer in Urdu/Arabic script; "
                    "if the target is Roman Urdu, use Latin letters only. Preserve all "
                    "facts, numbers, and source citations.",
                ),
            ]
        )
        correction_chain = correction_prompt | _get_llm()
        response = correction_chain.invoke(
            {**inputs, "draft_answer": response.content}
        )

    if target_language == "roman_urdu":
        response.content = clean_roman_urdu_vocabulary(response.content)

    if language_followup:
        sources = previous_answer.get("sources", [])
    else:
        sources = sources_used_by_answer(docs, response.content)
    return {"answer": strip_sources_line(response.content), "sources": sources}
