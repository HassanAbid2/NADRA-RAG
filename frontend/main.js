const examples = [
  "What documents are required to renew a CNIC?",
  "NICOP banwane ki fees aur process kya hai?",
  "How can I track my NADRA application?",
  "FRC ke liye kaun eligible hai?",
];

const elements = {
  examples: document.querySelector("#examples"),
  form: document.querySelector("#chat-form"),
  question: document.querySelector("#question"),
  send: document.querySelector("#send-button"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  hero: document.querySelector("#hero"),
  newChat: document.querySelector("#new-chat"),
  menu: document.querySelector("#menu-button"),
  sidebar: document.querySelector("#sidebar"),
  scrim: document.querySelector("#sidebar-scrim"),
};

let waiting = false;
let conversationHistory = [];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(value) {
  const safe = escapeHtml(value).replace(/\r\n/g, "\n");
  const blocks = safe.split(/\n{2,}/).map((block) => {
    const lines = block.split("\n");
    if (lines.every((line) => /^\s*[-*]\s+/.test(line))) {
      return `<ul>${lines
        .map((line) => `<li>${inlineMarkdown(line.replace(/^\s*[-*]\s+/, ""))}</li>`)
        .join("")}</ul>`;
    }
    if (lines.every((line) => /^\s*\d+[.)]\s+/.test(line))) {
      return `<ol>${lines
        .map((line) => `<li>${inlineMarkdown(line.replace(/^\s*\d+[.)]\s+/, ""))}</li>`)
        .join("")}</ol>`;
    }
    return `<p>${lines.map(inlineMarkdown).join("<br>")}</p>`;
  });
  return blocks.join("");
}

function inlineMarkdown(line) {
  return line
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function createSources(sources) {
  if (!sources?.length) return null;

  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `View sources (${sources.length})`;
  details.append(summary);

  const list = document.createElement("div");
  list.className = "source-list";
  sources.forEach((source) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.textContent = `Document · ${source.source || "Official document"} · Page ${source.page ?? "—"}`;
    list.append(item);
  });
  details.append(list);
  return details;
}

function addMessage(role, content, sources = []) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "YOU" : "PK";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.setAttribute("dir", "auto");
  bubble.innerHTML = renderMarkdown(content);
  const sourceList = createSources(sources);
  if (sourceList) bubble.append(sourceList);

  message.append(avatar, bubble);
  elements.messages.append(message);
  setConversationMode(true);
  message.scrollIntoView({ behavior: "smooth", block: "end" });
  return message;
}

function addTypingIndicator() {
  const message = document.createElement("article");
  message.className = "message assistant";
  message.setAttribute("aria-label", "NADRA Guide is preparing an answer");
  message.innerHTML = `
    <div class="avatar" aria-hidden="true">PK</div>
    <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>
  `;
  elements.messages.append(message);
  message.scrollIntoView({ behavior: "smooth", block: "end" });
  return message;
}

function setConversationMode(active) {
  document.body.classList.toggle("chat-active", active);
}

function setWaiting(value) {
  waiting = value;
  elements.question.disabled = value;
  elements.send.disabled = value;
  elements.form.classList.toggle("is-waiting", value);
}

function resizeTextarea() {
  elements.question.style.height = "auto";
  elements.question.style.height = `${Math.min(elements.question.scrollHeight, 150)}px`;
}

async function ask(question) {
  if (waiting || !question.trim()) return;

  const cleanQuestion = question.trim();
  const priorHistory = conversationHistory.slice(-12);
  addMessage("user", cleanQuestion);
  conversationHistory.push({ role: "user", content: cleanQuestion, sources: [] });
  elements.question.value = "";
  resizeTextarea();
  setWaiting(true);
  const typing = addTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: cleanQuestion,
        history: priorHistory,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "The assistant is temporarily unavailable.");
    }
    typing.remove();
    addMessage("assistant", payload.answer, payload.sources);
    conversationHistory.push({
      role: "assistant",
      content: payload.answer,
      sources: payload.sources || [],
    });
  } catch (error) {
    typing.remove();
    addMessage(
      "assistant",
      error.message || "I couldn't complete that request. Please check the API connection and try again.",
    );
  } finally {
    setWaiting(false);
    elements.question.focus();
  }
}

function setSidebar(open) {
  elements.sidebar.classList.toggle("open", open);
  elements.scrim.classList.toggle("open", open);
  elements.menu.setAttribute("aria-expanded", String(open));
}

examples.forEach((question) => {
  const button = document.createElement("button");
  button.className = "example-button";
  button.type = "button";
  button.textContent = question;
  button.style.setProperty("--order", examples.indexOf(question));
  button.addEventListener("click", () => {
    setSidebar(false);
    ask(question);
  });
  elements.examples.append(button);
});

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(elements.question.value);
});

elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

elements.question.addEventListener("input", resizeTextarea);
elements.newChat.addEventListener("click", () => {
  elements.messages.replaceChildren();
  conversationHistory = [];
  setConversationMode(false);
  setSidebar(false);
  elements.question.focus();
});
elements.menu.addEventListener("click", () => {
  setSidebar(!elements.sidebar.classList.contains("open"));
});
elements.scrim.addEventListener("click", () => setSidebar(false));

elements.question.focus();
