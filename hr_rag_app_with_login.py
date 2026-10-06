from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
POLICIES_FILE = BASE_DIR / "policies_approved.csv"
FAQ_FILE = BASE_DIR / "hr_faq_approved_final.csv"
CONTACT_FILE = BASE_DIR / "contacts (1).csv"
DB_FILE = BASE_DIR / "hr_rag.db"

ROLES = ["intern", "associate", "store_manager", "vp", "ceo", "hr"]
REGIONS = ["ALL"]

# Demo accounts for development/testing only. Change these before production.
DEMO_USERS = {
    "employee@nexahr.demo": {"password": "Employee@123", "role": "associate", "name": "Demo Employee"},
    "hr@nexahr.demo": {"password": "HR@12345", "role": "hr", "name": "Demo HR"},
    "ceo@nexahr.demo": {"password": "CEO@12345", "role": "ceo", "name": "Demo CEO"},
}
SENSITIVE_TERMS = {
    "harassment", "sexual harassment", "retaliation", "termination",
    "medical", "legal", "whistleblowing",
}
STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "get", "how",
    "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "to", "what", "when", "where", "which", "with", "you", "your",
    "employee", "employees", "day", "days", "much", "many", "please",
    "tell", "about", "would", "could", "should", "have", "has", "had",
}

st.set_page_config(
    page_title="NexaHR • AI Helpdesk",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def login_screen() -> bool:
    """Render a simple session login screen using demo credentials."""
    if st.session_state.get("authenticated", False):
        return True

    st.markdown(
        """
        <div class="login-wrap">
          <div class="login-card">
            <div class="hero-kicker">NexaHR · Secure Access</div>
            <div class="login-title">HR Helpdesk</div>
            <div class="login-sub">Sign in to access HR policies authorized for your role.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.15, 1])
    with center:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Work email", placeholder="you@company.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            account = DEMO_USERS.get(email.strip().lower())
            if account and password == account["password"]:
                st.session_state.authenticated = True
                st.session_state.user_email = email.strip().lower()
                st.session_state.user_name = account["name"]
                st.session_state.user_role = account["role"]
                st.session_state.user_region = "ALL"
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Invalid email or password.")

        with st.expander("Demo login credentials", expanded=True):
            st.caption("These accounts are for testing only — do not use them in production.")
            for email, account in DEMO_USERS.items():
                st.code(f"Email: {email}\nPassword: {account['password']}\nRole: {account['role']}")

    return False


def logout():
    for key in ["authenticated", "user_email", "user_name", "user_role", "user_region", "messages", "pending_query"]:
        st.session_state.pop(key, None)
    st.rerun()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def allowed_for_user(row: pd.Series, role: str, region: str) -> bool:
    roles = {x.strip().lower() for x in str(row.get("allowed_roles", "")).split(";") if x.strip()}
    regions = {x.strip().lower() for x in str(row.get("regions", "")).split(";") if x.strip()}
    return ("all" in roles or role.lower() in roles) and ("all" in regions or region.lower() in regions)


def is_sensitive(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in SENSITIVE_TERMS)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", normalize(text)) if t not in STOPWORDS}


def citation(row: pd.Series) -> str:
    return (
        f'{row["policy_name"]} §{row["section"]} '
        f'(v{row["version"]}, effective {row["effective_date"]})'
    )


def status_note(row: pd.Series) -> str:
    if str(row.get("version", "")).upper().endswith("DRAFT") or str(row.get("effective_date", "")) == "TO_BE_DEFINED_BY_MANAGEMENT":
        return (
            "Source metadata note: this row is marked APPROVED, but its version/effective-date "
            "fields still contain management placeholders."
        )
    return ""


def safe(val: Any) -> str:
    return str(val or "").strip()


class HRRAG:
    """Hybrid retrieval-augmented generation engine over the supplied HR CSV knowledge base."""

    def __init__(self):
        self.policies = load_csv(POLICIES_FILE)
        self.faq = load_csv(FAQ_FILE)
        self.contacts = load_csv(CONTACT_FILE)

        faq_grouped = (
            self.faq.groupby("chunk_id", as_index=False)
            .agg(
                faq_questions=("question", lambda s: " | ".join(list(dict.fromkeys(map(str, s)))[:10])),
                faq_answers=("answer", lambda s: " | ".join(list(dict.fromkeys(map(str, s)))[:3])),
                faq_intents=("intent", lambda s: " | ".join(list(dict.fromkeys(map(str, s)))[:10])),
                faq_category=("category", "first"),
                faq_form_name=("form_name", "first"),
                faq_form_link=("form_link", "first"),
                faq_contact_id=("contact_id", "first"),
            )
        )
        self.chunks = self.policies.merge(faq_grouped, on="chunk_id", how="left").fillna("")
        for c in self.chunks.columns:
            self.chunks[c] = self.chunks[c].astype(str)

        self.chunks["search_text"] = (
            self.chunks["policy_name"] + " " +
            self.chunks["section_title"] + " " +
            self.chunks["category"] + " " +
            self.chunks["key_rule"] + " " +
            self.chunks["text"] + " " +
            self.chunks["faq_intents"] + " " +
            self.chunks["faq_questions"] + " " +
            self.chunks["faq_answers"]
        )

        # Word + character retrieval helps with both natural questions and short/typo-heavy queries.
        self.word_vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 3), sublinear_tf=True,
            min_df=1, stop_words=list(STOPWORDS), norm="l2"
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            min_df=1, norm="l2"
        )
        self.word_matrix = self.word_vec.fit_transform(self.chunks["search_text"])
        self.char_matrix = self.char_vec.fit_transform(self.chunks["search_text"])
        self.contact_map = {r["contact_id"]: r.to_dict() for _, r in self.contacts.iterrows()}

    @property
    def stats(self) -> dict[str, int]:
        return {
            "policies": len(self.policies),
            "faqs": len(self.faq),
            "chunks": len(self.chunks),
            "contacts": len(self.contacts),
        }

    def _exact_boost(self, query: str, row: pd.Series) -> float:
        q = normalize(query)
        combined = normalize(" ".join([
            row["policy_name"], row["section_title"], row["category"], row["key_rule"],
            row["text"], row["faq_intents"], row["faq_questions"]
        ]))
        faq_questions = normalize(row["faq_questions"])
        q_tokens = tokens(q)
        doc_tokens = tokens(combined)
        overlap = len(q_tokens & doc_tokens) / max(1, len(q_tokens))

        boost = 0.30 * overlap
        if q and q in faq_questions:
            boost += 0.55
        important_phrases = [
            "annual leave", "sick leave", "parental leave", "unpaid leave", "bereavement leave",
            "carry forward", "carry over", "half day", "medical certificate", "urgent leave",
            "leave balance", "salary impact", "notice period", "apply for leave", "leave request",
            "take leave", "use leave", "leave entitlement", "leave eligibility",
            "probation", "work hours", "attendance", "business travel", "reimbursement",
        ]
        for phrase in important_phrases:
            if phrase in q and phrase in combined:
                boost += 0.22 if len(phrase.split()) >= 2 else 0.12
        # Prefer a direct category match without over-rewarding generic words like "leave".
        if normalize(row["category"]) in q or normalize(row["section_title"]) in q:
            boost += 0.18
        return min(boost, 0.90)

    def _expand_query(self, query: str) -> str:
        """Add lightweight policy-domain synonyms so natural employee questions retrieve the right policy.

        This is retrieval-only expansion: it does not create policy facts and therefore does not
        weaken the approved-KB grounding requirement.
        """
        q = normalize(query)
        additions: list[str] = []

        # Leave questions are often phrased more casually than the policy/FAQ wording.
        if re.search(r"\bleave\b", q):
            if re.search(r"\b(can|may|could|should|take|use|get|have|eligible|entitled)\b", q):
                additions += [
                    "annual leave", "general leave", "leave entitlement",
                    "leave eligibility", "paid leave", "leave request"
                ]
            if re.search(r"\b(request|apply|put in|book)\b", q):
                additions += ["annual leave request", "apply for leave", "HR portal"]
            if re.search(r"\b(carry|roll|unused)\b", q):
                additions += ["carry forward", "unused annual leave"]

        # Common HR phrasing variants.
        mappings = {
            "salary": ["pay", "payroll", "compensation"],
            "quit": ["resignation", "notice period", "separation"],
            "fire": ["termination", "discipline"],
            "late": ["attendance", "punctuality", "late arrival"],
            "insurance": ["health insurance", "medical benefit", "employee benefit"],
        }
        for term, synonyms in mappings.items():
            if re.search(rf"\b{re.escape(term)}\b", q):
                additions.extend(synonyms)

        return q + (" " + " ".join(dict.fromkeys(additions)) if additions else "")

    def _gemini_query_rewrite(self, query: str, role: str, region: str) -> str:
        """Use Gemini to improve retrieval wording. Failure is non-fatal."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return ""
        model = os.getenv("GEMINI_MODEL", "gemini-2.8-flash").strip()
        instruction = (
            "You are an HR policy search-query optimizer. Rewrite the employee question "
            "into a short list of policy search concepts. Do not answer the question. "
            "Do not invent company-specific facts. Return ONLY plain-text search terms. "
            f"Employee role: {role}. Region: {region}."
        )
        try:
            response = self._gemini_rest_generate(
                model=model,
                system_instruction=instruction,
                prompt=f"Employee question: {query}",
                max_output_tokens=100,
                temperature=0,
            )
            return re.sub(r"[\r\n]+", " ", response).strip()[:500]
        except Exception:
            return ""

    def _gemini_rest_generate(
        self, model: str, system_instruction: str, prompt: str,
        max_output_tokens: int = 700, temperature: float = 0.2
    ) -> str:
        """Call Gemini directly over its documented REST API. This avoids SDK/version mismatches."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        response = requests.post(
            url, params={"key": api_key}, json=payload, timeout=45
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {detail}")
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if not text:
            finish = candidates[0].get("finishReason", "unknown")
            raise RuntimeError(f"Gemini returned no text (finish reason: {finish}).")
        return text

    def retrieve(self, query: str, role: str, region: str, k: int = 5) -> list[dict[str, Any]]:
        mask = (
            self.chunks["access_status"].str.upper().eq("APPROVED") &
            self.chunks.apply(lambda r: allowed_for_user(r, role, region), axis=1)
        )
        idxs = np.flatnonzero(mask.to_numpy())
        if len(idxs) == 0:
            return []

        retrieval_query = self._expand_query(query)
        semantic_query = self._gemini_query_rewrite(query, role, region)
        if semantic_query:
            retrieval_query = f"{retrieval_query} {semantic_query}"
        wq = self.word_vec.transform([retrieval_query])
        cq = self.char_vec.transform([retrieval_query])
        word_scores = cosine_similarity(wq, self.word_matrix[idxs]).ravel()
        char_scores = cosine_similarity(cq, self.char_matrix[idxs]).ravel()

        rows = []
        for local_i, idx in enumerate(idxs):
            row = self.chunks.iloc[idx]
            exact = self._exact_boost(query, row)
            hybrid = 0.58 * float(word_scores[local_i]) + 0.22 * float(char_scores[local_i]) + 0.20 * exact
            rows.append((idx, hybrid, float(word_scores[local_i]), float(char_scores[local_i]), exact))

        rows.sort(key=lambda x: x[1], reverse=True)
        out: list[dict[str, Any]] = []
        seen = set()
        for idx, score, ws, cs, ex in rows:
            row = self.chunks.iloc[idx]
            if row["chunk_id"] in seen or score <= 0:
                continue
            seen.add(row["chunk_id"])
            out.append({"row": row, "score": score, "word": ws, "char": cs, "exact": ex})
            if len(out) >= k:
                break
        return out

    def confidence(self, results: list[dict[str, Any]], sensitive: bool = False) -> dict[str, Any]:
        if not results:
            return {"score": 0.0, "label": "No match", "color": "#ef4444", "reason": "No authorized source matched the question."}
        top = results[0]["score"]
        second = results[1]["score"] if len(results) > 1 else 0.0
        margin = max(0.0, top - second)
        # Convert retrieval strength + ranking separation into a bounded confidence indicator.
        strength = min(1.0, top / 0.75)
        separation = min(1.0, margin / 0.20)
        score = 0.72 * strength + 0.28 * separation
        if top < (0.24 if sensitive else 0.20):
            score *= 0.55
        score = max(0.0, min(1.0, score))
        if score >= 0.78:
            label, color = "High", "#22c55e"
        elif score >= 0.55:
            label, color = "Medium", "#f59e0b"
        else:
            label, color = "Low", "#ef4444"
        reason = f"Top retrieval {top:.3f}; ranking margin {margin:.3f}."
        return {"score": score, "label": label, "color": color, "reason": reason}

    def context_for_generation(self, results: list[dict[str, Any]]) -> str:
        blocks = []
        for i, item in enumerate(results[:4], start=1):
            r = item["row"]
            blocks.append(
                f"SOURCE {i}\n"
                f"Chunk ID: {r['chunk_id']}\n"
                f"Policy: {r['policy_name']} §{r['section']} — {r['section_title']}\n"
                f"Category: {r['category']}\n"
                f"Rule: {r['key_rule']}\n"
                f"Policy text: {r['text']}\n"
                f"Approved FAQ answer(s): {r['faq_answers']}\n"
                f"Form: {r['form_name']}\n"
                f"Source status: {r['access_status']}\n"
                f"Version: {r['version']}\n"
                f"Effective date: {r['effective_date']}"
            )
        return "\n\n---\n\n".join(blocks)

    def generate_with_gemini(self, query: str, results: list[dict[str, Any]], role: str, region: str) -> tuple[str | None, str | None]:
        """Generate a grounded answer with Gemini. Uses direct REST and a small model fallback chain."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None, "GEMINI_API_KEY is not configured."

        system = (
            "You are NexaHR, a careful internal HR helpdesk assistant. "
            "Answer ONLY from the supplied authorized company policy sources. "
            "Do not invent policies, dates, numbers, benefits, exceptions, contacts, "
            "approval authorities, disciplinary consequences, or links. "
            "If the supplied sources do not answer the question, explicitly say that "
            "the policy information is not specified and direct the employee to HR Helpdesk. "
            "Respect the employee role and region. Prefer the most specific applicable source. "
            "Keep answers concise, practical, and professional. Mention the policy section(s) used. "
            f"Employee role: {role}. Region: {region}."
        )
        prompt = (
            f"Employee question: {query}\n\n"
            f"Authorized knowledge base context:\n{self.context_for_generation(results)}\n\n"
            "Write the answer using only the authorized context above. "
            "If multiple sources are relevant, reconcile them without inventing facts."
        )

        configured = os.getenv("GEMINI_MODEL", "gemini-2.8-flash").strip()
        models = [configured] if configured else []
        errors = []
        for model in models:
            try:
                text = self._gemini_rest_generate(
                    model=model, system_instruction=system, prompt=prompt,
                    max_output_tokens=700, temperature=0.2
                )
                return text, None
            except Exception as exc:
                errors.append(f"{model}: {str(exc)}")
                # Try the next compatible model only for availability/access errors.
                if not any(code in str(exc) for code in ["HTTP 404", "HTTP 403", "HTTP 429", "HTTP 400"]):
                    break
        message = " | ".join(errors)
        if len(message) > 700:
            message = message[:697] + "..."
        return None, message

    def answer(self, query: str, role: str, region: str, use_gemini: bool = True) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"answer": "Please enter an HR question.", "sources": [], "confidence": self.confidence([]), "sensitive": False}

        sensitive = is_sensitive(query)
        results = self.retrieve(query, role, region, k=5)
        conf = self.confidence(results, sensitive=sensitive)

        if not results:
            return {
                "answer": "I couldn't find an approved, authorized HR source that answers this question for your current role and region. Please contact HR Helpdesk.",
                "sources": [], "confidence": conf, "sensitive": sensitive,
            }

        top = results[0]
        top_row = top["row"]

        # Gemini is allowed to interpret natural language and decide whether the retrieved
        # evidence actually answers the question. The retrieval score is a diagnostic, not
        # a hard gate: otherwise conversational questions such as "can I take a leave?"
        # can be rejected before the LLM ever sees the approved policy evidence.
        generated = None
        generation_error = None
        if use_gemini:
            generated, generation_error = self.generate_with_gemini(query, results, role, region)

        if generated:
            answer = generated
        else:
            # When Gemini is unavailable, retain a conservative local-RAG gate.
            threshold = 0.24 if sensitive else 0.20
            if top["score"] < threshold:
                return {
                    "answer": "I don't have a sufficiently strong approved match for that question. Please rephrase it or contact HR Helpdesk.",
                    "sources": results[:3], "confidence": conf, "sensitive": sensitive,
                    "mode": "Local RAG", "generation_error": generation_error,
                }
            answer = safe(top_row["faq_answers"]) or safe(top_row["text"])

        extras = []
        if safe(top_row["faq_form_name"] or top_row["form_name"]):
            extras.append(f"**Form:** {safe(top_row['faq_form_name'] or top_row['form_name'])}")
        contact = self.contact_map.get(top_row["contact_id"])
        if contact is not None:
            team = safe(contact.get("team"))
            email = safe(contact.get("email"))
            phones = safe(contact.get("phones"))
            contact_bits = [x for x in [team, email, phones] if x and x != "TO_BE_DEFINED_BY_MANAGEMENT"]
            if contact_bits:
                extras.append("**Contact:** " + " | ".join(contact_bits))
        note = status_note(top_row)
        if note:
            extras.append(f"_**Data note:** {note}_")
        if extras:
            answer += "\n\n" + "\n".join(extras)

        return {
            "answer": answer,
            "sources": results[:4],
            "confidence": conf,
            "sensitive": sensitive,
            "mode": "Gemini RAG" if generated else "Local RAG",
            "generation_error": generation_error,
        }


def build_database() -> None:
    policies = load_csv(POLICIES_FILE)
    faq = load_csv(FAQ_FILE)
    sections = load_csv(BASE_DIR / "hr_policy_sections (1).csv")
    contacts = load_csv(CONTACT_FILE)
    with sqlite3.connect(DB_FILE) as con:
        policies.to_sql("policies", con, if_exists="replace", index=False)
        faq.to_sql("faqs", con, if_exists="replace", index=False)
        sections.to_sql("policy_sections", con, if_exists="replace", index=False)
        contacts.to_sql("contacts", con, if_exists="replace", index=False)


@st.cache_resource(show_spinner=False)
def load_rag() -> HRRAG:
    return HRRAG()


def inject_css():
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(99,102,241,.16), transparent 28%),
            radial-gradient(circle at 90% 12%, rgba(14,165,233,.13), transparent 24%),
            #080b13;
        color: #eef2ff;
    }
    .block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1321 0%, #0a0d16 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    .hero {
        padding: 26px 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(99,102,241,.20), rgba(14,165,233,.08) 55%, rgba(255,255,255,.03));
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 20px 70px rgba(0,0,0,.28);
        margin-bottom: 22px;
    }
    .hero-kicker {font-size: .82rem; letter-spacing: .16em; text-transform: uppercase; color: #a5b4fc; font-weight: 700;}
    .hero-title {font-size: 2.35rem; font-weight: 800; margin: 5px 0 3px; letter-spacing: -.03em;}
    .hero-sub {color: #b8c1d1; font-size: 1rem;}
    .metric {
        padding: 14px 16px; border-radius: 16px;
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08);
    }
    .metric-num {font-size: 1.45rem; font-weight: 800;}
    .metric-label {font-size: .78rem; color: #98a3b7;}
    .context-chip {display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(99,102,241,.13); color:#c7d2fe; margin-right:6px; font-size:.78rem; border:1px solid rgba(129,140,248,.2);}
    .source-card {padding: 14px 16px; border-radius: 16px; background: rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.07); margin-bottom: 10px;}
    .source-head {font-weight:700; color:#e5e7eb;}
    .source-meta {font-size:.78rem; color:#98a3b7; margin-top:4px;}
    .confidence-wrap {padding:16px; border-radius:18px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);}
    .confidence-label {font-size:.8rem; color:#9ca3af; text-transform:uppercase; letter-spacing:.08em;}
    .confidence-score {font-size:1.8rem; font-weight:800;}
    .confidence-bar {height:10px; background:#1f2937; border-radius:999px; overflow:hidden; margin:10px 0 8px;}
    .confidence-fill {height:100%; border-radius:999px;}
    .tiny {font-size:.76rem; color:#8d98ab;}
    div[data-testid="stChatMessage"] {border-radius: 20px; border: 1px solid rgba(255,255,255,.05);}
    div[data-testid="stChatInput"] {border:1px solid rgba(129,140,248,.25); border-radius:18px;}
    .login-wrap {max-width:760px; margin:7vh auto 18px;}
    .login-card {text-align:center; padding:34px 30px 20px; border-radius:24px; background:linear-gradient(135deg, rgba(99,102,241,.20), rgba(14,165,233,.08)); border:1px solid rgba(148,163,184,.18); box-shadow:0 20px 70px rgba(0,0,0,.28);}
    .login-title {font-size:2.4rem; font-weight:800; margin-top:6px;}
    .login-sub {color:#b8c1d1; margin-top:6px;}
    .stButton>button {border-radius:12px;}
    </style>
    """, unsafe_allow_html=True)


def confidence_card(conf: dict[str, Any]):
    pct = int(round(conf["score"] * 100))
    color = conf["color"]
    st.markdown(
        f"""
        <div class="confidence-wrap">
          <div class="confidence-label">Retrieval confidence</div>
          <div class="confidence-score">{pct}% <span style="font-size:.95rem;color:{color};">{conf['label']}</span></div>
          <div class="confidence-bar"><div class="confidence-fill" style="width:{pct}%;background:{color};"></div></div>
          <div class="tiny">{conf['reason']} This is a retrieval-confidence indicator, not a guarantee of answer correctness.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_cards(sources: list[dict[str, Any]]):
    st.markdown("### Retrieved sources")
    for item in sources:
        r = item["row"]
        pct = int(round(min(1.0, item["score"]) * 100))
        st.markdown(
            f"""
            <div class="source-card">
              <div class="source-head">{r['policy_name']} · §{r['section']} · {r['section_title']}</div>
              <div class="source-meta">Chunk {r['chunk_id']} · retrieval {pct}% · category {r['category']}</div>
              <div style="margin-top:8px;color:#cbd5e1;">{r['key_rule']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    inject_css()
    if not login_screen():
        return
    kb = load_rag()

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">NexaHR · Knowledge Grounded AI</div>
          <div class="hero-title">HR Helpdesk <span style="color:#818cf8;">RAG</span></div>
          <div class="hero-sub">Ask natural-language HR questions. The system retrieves authorized policy evidence first, then answers only from that evidence.</div>
          <div style="margin-top:13px;">
            <span class="context-chip">Role-aware</span>
            <span class="context-chip">Region-aware</span>
            <span class="context-chip">Approved KB only</span>
            <span class="context-chip">Confidence scoring</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Your access")
        st.markdown(f"**{st.session_state.get('user_name', 'Employee')}**")
        st.caption(st.session_state.get("user_email", ""))
        role = st.session_state.get("user_role", "associate")
        region = st.session_state.get("user_region", "ALL")
        st.caption(f"Role: `{role}` · Region: `{region}`")
        if st.button("Log out", use_container_width=True):
            logout()
        st.divider()
        use_gemini = st.toggle("Use Gemini for response writing", value=True,
                               help="Requires GEMINI_API_KEY. Retrieval and access filtering always happen first.")
        st.divider()
        st.markdown("### Knowledge base")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Policy chunks", kb.stats["chunks"])
            st.metric("FAQs", kb.stats["faqs"])
        with c2:
            st.metric("Policies", kb.stats["policies"])
            st.metric("Contacts", kb.stats["contacts"])
        st.caption("Only rows marked APPROVED and authorized for your selected role/region are eligible for retrieval.")
        if use_gemini and not os.getenv("GEMINI_API_KEY", "").strip():
            st.warning("Gemini is enabled, but GEMINI_API_KEY is not configured. The app will fall back to local RAG answers.")
        if use_gemini and os.getenv("GEMINI_API_KEY", "").strip():
            st.caption(f"Gemini model: {os.getenv('GEMINI_MODEL', 'gemini-3.5-flash')}")
        if st.button("Rebuild SQLite database", use_container_width=True):
            build_database()
            st.success("Database rebuilt.")
        st.caption("💡 Tip: ask specific questions such as “Can I carry forward unused annual leave?”")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        left, right = st.columns([1.65, 1])
        with left:
            st.markdown("### Start with a question")
            suggestions = [
                "How many annual leave days do I get?",
                "Can I carry forward unused annual leave?",
                "How do I request annual leave?",
                "What is the sick leave policy?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(s, key=f"suggest_{i}", use_container_width=True):
                        st.session_state.pending_query = s
                        st.rerun()
        with right:
            st.markdown("### How it works")
            st.write("**1. Retrieve** authorized evidence from the HR database.")
            st.write("**2. Rank** sources using hybrid lexical retrieval.")
            st.write("**3. Answer** from the strongest approved evidence.")
            st.write("**4. Explain** confidence and show the retrieved sources.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("confidence"):
                conf = msg["confidence"]
                pct = int(round(conf["score"] * 100))
                color = conf["color"]
                st.markdown(
                    f'<div class="tiny">Retrieval confidence: <span style="color:{color};font-weight:700;">{pct}% · {conf["label"]}</span></div>',
                    unsafe_allow_html=True,
                )

    pending = st.session_state.pop("pending_query", None)
    q = st.chat_input("Ask an HR policy question…")
    q = q or pending
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving authorized HR evidence…"):
                result = kb.answer(q, role, region, use_gemini=use_gemini)
            st.markdown(result["answer"])
            confidence_card(result["confidence"])
            if result.get("sources"):
                with st.expander("View retrieved evidence", expanded=False):
                    source_cards(result["sources"])
            if result.get("sensitive"):
                st.warning("Sensitive HR topic detected. The response is restricted to directly supported approved evidence.")
            if result.get("generation_error") and use_gemini:
                st.warning("Gemini response generation was unavailable. The app returned the approved local knowledge-base answer instead.")
                if result.get("generation_error"):
                    st.error(f"Gemini API error: {result['generation_error']}")
            mode = result.get("mode")
            if mode:
                st.caption(f"Response mode: {mode}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "confidence": result["confidence"],
        })


if __name__ == "__main__":
    build_database()
    main()
