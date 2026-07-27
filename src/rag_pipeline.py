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
TOP_K = 6

SYSTEM_PROMPT = """You are an official FAQ assistant for NADRA (National Database and \
Registration Authority, Pakistan). You answer citizen questions about NADRA services: \
CNIC, NICOP, POC, CRC (Child Registration Certificate), and FRC (Family Registration \
Certificate) — covering required documents, eligibility, fees, processing timelines, \
and procedures.

STRICT RULES:
1. Answer ONLY using the CONTEXT provided below. Never use outside knowledge.
2. If the context contains no relevant information at all, do not guess. Say that \
verified NADRA information is unavailable and advise the user to contact the NADRA \
helpline (1777) or visit www.nadra.gov.pk, using the language rules below. For an \
English question, reply exactly: "I don't have verified NADRA information to answer \
that. Please contact the NADRA helpline (1777) or visit www.nadra.gov.pk."
2b. If the context answers the question only partially, give the part that IS \
covered and clearly say which part is not covered by official NADRA documents.
2c. Use CONVERSATION HISTORY to understand follow-up questions. If the user asks \
to repeat, translate, shorten, or change the language of the previous answer, act \
on the previous answer instead of treating the request as a new NADRA topic.
2d. Read the INTERPRETED CURRENT QUESTION carefully and directly answer every part \
the user asked. For a procedure, give clear numbered steps in the order supported by \
the context. For documents, fees, eligibility, or timelines, use separate labeled \
sections when the question asks for more than one. Ignore retrieved passages that are \
not relevant to the interpreted question.
2e. Answer only what the user asked. Do not append fee, timeline, document, eligibility, \
or other sections unless the user requested them or they are essential to the requested \
procedure.
2f. Never invent or generalize a document name. If a guide says "upload necessary \
documents" without naming them, state that the guide does not specify the document list. \
If another relevant policy passage gives precise resident/non-resident requirements, \
state those requirements exactly instead of saying "other required documents."
2g. For an app procedure, include the starting navigation path from the relevant guide \
(for example, the app section and option to tap) before later steps. Do not skip the \
entry point.
3. Match the user's language and writing style:
   - English question: answer in clear, plain English.
   - Urdu-script question: answer in Pakistani Urdu using the Urdu/Arabic script.
   - Roman-Urdu or mixed Urdu-English question: answer in natural Pakistani Roman \
     Urdu using the Latin alphabet, keeping familiar English service names and \
     technical terms where helpful.
   - If uncertain, mirror the language and script used by the user. Do not answer an \
     Urdu or Roman-Urdu question only in English.
   - NEVER answer in Hindi or use Devanagari characters. Urdu and Hindi are not \
     interchangeable for this assistant.
4. When the context gives fees, timelines, or document lists, state them precisely as \
written — do not round or alter numbers while translating the surrounding explanation.
5. End your answer with a "Sources:" line listing the document names and pages you used. \
For a language-only rewrite, retain the previous answer's source citations.

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
}
_LANGUAGE_FOLLOWUP_RE = re.compile(
    r"\b(?:reply|answer|respond|translate|write|likho|batao)\b.*"
    r"\b(?:urdu|roman\s+urdu|english)\b|"
    r"\b(?:urdu|roman\s+urdu|english)\b.*\b(?:reply|answer|mein|main|me)\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^\s*(?:a+o+a+|aoa|ass?alam(?:\s+o|\s+u|\s+alaikum)?|"
    r"السلام\s+علیکم|سلام|hello|hi|hey)\s*[!.?]*\s*$",
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
    if re.search(r"\b(?:reply|answer|respond|write)\s+in\s+urdu\b", lowered):
        return "urdu"
    if re.search(r"\burdu\s+(?:mein|main|me)\b", lowered):
        return "urdu"

    tokens = set(re.findall(r"[a-z]+", lowered))
    marker_count = len(tokens & _ROMAN_URDU_MARKERS)
    if marker_count >= 2 or tokens & {"banwana", "banwane", "chahiye"}:
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
        r"\b(?:aoa|ass?alam)\b", question.lower()
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
            for page in dict.fromkeys(cited_pages):
                if page in available_pages:
                    sources.append({"source": source, "page": page})
        else:
            sources.append({"source": source, "page": None})
    return sources


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
    return {"answer": response.content, "sources": sources}
