# app.py

import streamlit as st
import time
import tempfile
import os
import re
import json
import hashlib
import secrets
import hmac
import html
import pandas as pd
from pathlib import Path
from src.rag_pipeline import RAGPipeline
from src.exam_intelligence import clean_question_display
try:  # REPEATED_SOLUTIONS_PATCH
    from repeated_solutions import lookup_solution as _rs_lookup, clean_for_export as _rs_clean
except Exception:
    def _rs_lookup(subject, question):
        return None
    def _rs_clean(text, pdf=False):
        return text
from src.curriculum import _LAB_RE
from src.document_loader import ocr_page, MIN_TEXT_LENGTH, extract_questions_from_text, extract_subject, extract_semester
from config import UNIVERSITY_NAME, DEPARTMENT, NOTES_DIR, PAPERS_DIR, SYLLABUS_DIR, ADMIN_PASSWORD, BASE_DIR
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
import pytesseract
from config import TESSERACT_CMD
if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# --- Student accounts ---
# Self-registered accounts (open signup — no admin needed to create one).
# Passwords are hashed (PBKDF2-HMAC-SHA256, stdlib only, no new dependency)
# — appropriate security for an offline single-laptop deployment, not a
# production system exposed to the internet: no rate limiting, no email
# verification, no password-reset flow.
STUDENTS_FILE = BASE_DIR / "students.json"


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100_000).hex()


class StudentFileUnreadable(Exception):
    """students.json exists but could not be understood. Never overwrite it in this state."""


def _read_students_file():
    """Returns (data, ok). ok=False means the file exists but could not be read or understood."""
    try:
        raw = STUDENTS_FILE.read_bytes()
    except OSError:
        return {}, False                      # locked by another program, permissions, ...
    for enc in ("utf-8-sig", "utf-16"):       # utf-16 = what PowerShell's ">" writes
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if not text.strip():
            return {}, True                   # a blank file is just "no students yet"
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data, True
    return {}, False


def _load_students(strict=False):
    if not STUDENTS_FILE.exists():
        return {}
    data, ok = _read_students_file()
    if ok:
        return data
    # SAFETY_FIX: keep a copy so a later save cannot destroy it
    try:
        import shutil as _sh
        _bk = str(STUDENTS_FILE) + ".corrupt"   # one backup, not one per call
        if not os.path.exists(_bk):
            _sh.copy2(STUDENTS_FILE, _bk)
    except Exception:
        pass
    if strict:
        raise StudentFileUnreadable("students.json could not be read")
    return {}


def _save_students(students):
    _tmp = str(STUDENTS_FILE) + ".tmp"
    with open(_tmp, 'w', encoding='utf-8') as f:
        json.dump(students, f, indent=2)
    os.replace(_tmp, STUDENTS_FILE)


def _register_student(email, name, password):
    students = _load_students(strict=True)   # refuse to save over an unreadable file
    salt = secrets.token_hex(16)
    students[email.lower().strip()] = {
        "name": name.strip(),
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_students(students)


def _verify_student(email, password):
    students = _load_students()
    record = students.get(email.lower().strip())
    if not record:
        return False
    return hmac.compare_digest(_hash_password(password, record["salt"]), str(record["password_hash"]))


# Login lockout: 5 wrong tries -> 60 s wait. Kept per-process (survives page refresh).
_MAX_FAILS = 5
_LOCK_SECONDS = 60


@st.cache_resource
def _login_attempts():
    return {}


def _login_locked(key):
    """Seconds left in the lockout for this key, or 0 if not locked."""
    rec = _login_attempts().get(key)
    if rec and rec["fails"] >= _MAX_FAILS:
        left = int(rec["last"] + _LOCK_SECONDS - time.time())
        if left > 0:
            return left
        _login_attempts().pop(key, None)
    return 0


def _login_failed(key):
    rec = _login_attempts().setdefault(key, {"fails": 0, "last": 0.0})
    rec["fails"] += 1
    rec["last"] = time.time()


def _login_reset(key):
    _login_attempts().pop(key, None)


def _reset_user_session():  # LOGIN_SESSION_PATCH
    """Forget everything that belongs to the previous user (chat, last topic, cached answers, ...).
    Login lockout counters live in cache_resource, so they are NOT affected."""
    for _k in list(st.session_state.keys()):
        del st.session_state[_k]

# No manual marks selector in the UI anymore — chat questions use this
# fixed default depth; question-paper answers still use the ACTUAL marks
# printed on each question (extracted via regex from the PDF itself, see
# extract_questions_from_pdf) — that part was never about the UI selector.
DEFAULT_CHAT_MARKS = 5

# PAGE CONFIG
st.set_page_config(
    page_title="uniRAG - Study Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------------------------
# STYLING (frontend only) — mobile-friendly spacing/typography, rounded chat
# bubbles, subtle sidebar background. Colours use rgba so they work in both
# Streamlit's light and dark themes.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* =====================================================================
   uniRAG — premium styling (CSS only). Page = soft icy grey-blue,
   cards = pure white floating above it, tabs = pill buttons.
   ===================================================================== */

/* ---------- Canvas & layout ---------- */
.stApp {
    background-color: #eef2f6;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    max-width: 1250px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #0f172a 40%, #1e3a8a 100%);
    border-right: none;
    box-shadow: 4px 0 18px rgba(15, 23, 42, 0.25);
}
/* sidebar: light text, glass buttons, soft dividers */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stToggle"] p { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.16) !important; }
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.22);
    color: #f1f5f9;
    box-shadow: none;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(96, 165, 250, 0.25);
    border-color: #93c5fd;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}
/* keep the glass look after a click (Streamlit leaves the button focused, which turned it white) */
[data-testid="stSidebar"] .stButton button:focus,
[data-testid="stSidebar"] .stButton button:focus-visible,
[data-testid="stSidebar"] .stButton button:active {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.22) !important;
    color: #f1f5f9 !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stButton button:focus:hover,
[data-testid="stSidebar"] .stButton button:active:hover {
    background: rgba(96, 165, 250, 0.25) !important;
    border-color: #93c5fd !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton button p { color: inherit !important; }
[data-testid="stSidebar"] svg { color: #cbd5e1; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { color: #93c5fd; }
[data-testid="stSidebarUserContent"] { padding-top: 0.5rem !important; }
[data-testid="stSidebar"] .unirag-side-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: #475569; margin: 4px 0 4px 2px;
}

/* Captions: Streamlit fades them with opacity, so colour AND opacity are set */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: #475569 !important;
    opacity: 1 !important;
}

/* ---------- Hero banner (unchanged) ---------- */
.unirag-hero-banner {
    background: linear-gradient(135deg, #090d16 0%, #0f172a 45%, #1e3a8a 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
    color: #ffffff;
}
.unirag-project-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59, 130, 246, 0.18);
    border: 1px solid rgba(147, 197, 253, 0.35);
    color: #93c5fd;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-bottom: 10px;
}
.unirag-hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.unirag-hero-title span { color: #60a5fa; font-weight: 600; }
.unirag-hero-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1;
    margin: 0;
    line-height: 1.5;
    font-weight: 400;
}
.unirag-hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.16);
    color: #f1f5f9;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    backdrop-filter: blur(8px);
}
.unirag-crest-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
    border-radius: 12px;
    color: #ffffff;
    font-size: 24px;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
}
/* compact one-row app banner */
.unirag-hero-compact { padding: 16px 22px !important; margin-bottom: 14px !important; }
.unirag-hero-compact .unirag-hero-title { font-size: 1.5rem !important; margin: 0 !important; }
.unirag-hero-compact .unirag-hero-subtitle { font-size: 0.86rem !important; margin-top: 4px !important; }
.unirag-hero-compact .unirag-project-pill { margin: 0 0 0 4px !important; padding: 2px 10px !important; font-size: 0.68rem !important; }
.unirag-hero-compact .unirag-crest-badge { width: 42px; height: 42px; font-size: 21px; flex: none; }
.unirag-hero-compact .unirag-hero-chip { font-size: 0.78rem; padding: 4px 11px; }

/* ---------- Tabs as pills / cards ---------- */
div[data-testid="stTabs"] { margin-top: 0.5rem; }
[data-baseweb="tab-list"],
div[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 6px 4px 10px 4px !important;   /* room so pill shadows are not clipped */
    gap: 10px !important;
    margin-bottom: 14px !important;
}
[data-baseweb="tab"],
div[data-testid="stTabs"] [role="tab"] {
    background: #ffffff !important;
    border: 1px solid #d5dde8 !important;
    border-radius: 12px !important;
    padding: 10px 22px !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
    transition: all 0.2s ease !important;
}
[data-baseweb="tab"]:hover,
div[data-testid="stTabs"] [role="tab"]:hover {
    color: #1e3a8a !important;
    border-color: #93c5fd !important;
    background: #f8fbff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.12) !important;
}
[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #2563eb !important;
    border: 1px solid #1d4ed8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    transform: none;
}
[data-baseweb="tab"][aria-selected="true"] *,
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] * { color: #ffffff !important; }
[data-baseweb="tab-list"]::after, [data-baseweb="tab-list"]::before { display: none !important; }
[data-baseweb="tabs"] { border-bottom: none !important; }
[data-baseweb="tab-list"] > div[role="presentation"],
[data-baseweb="tabs"] > div[role="presentation"] { display: none !important; }
/* hide EVERYTHING inside the tab bar that is not a tab: the blue underline (highlight) and the grey line (border) */
[role="tablist"] > *:not([role="tab"]),
[data-baseweb="tab-list"] > *:not([data-baseweb="tab"]):not([role="tab"]) {
    display: none !important;
    height: 0 !important;
    border: none !important;
    background: transparent !important;
}
/* ...and any underline drawn as a pseudo-element or bottom border on the tabs themselves */
[role="tab"]::before, [role="tab"]::after,
[data-baseweb="tab"]::before, [data-baseweb="tab"]::after {
    content: none !important;
    display: none !important;
}
[role="tablist"] { border-bottom: none !important; box-shadow: none !important; }
/* the underline + grey line are empty <div>s (anywhere inside the tab bar) — hide every empty div there */
[role="tablist"] div:empty,
[data-baseweb="tab-list"] div:empty,
[data-baseweb="tabs"] > div:empty,
[data-testid="stTabs"] [role="tablist"] ~ div:empty {
    display: none !important; height: 0 !important; width: 0 !important; background: transparent !important;
}
/* Streamlit can also draw the underline as a bottom border / shadow on the tab or its parent */
[role="tab"], [role="tab"][aria-selected="true"], [role="tab"]:hover {
    border-bottom: 1px solid transparent !important;
    text-decoration: none !important;
}
[role="tab"][aria-selected="true"] { border-bottom-color: #1d4ed8 !important; }
[role="tablist"] > div, [data-baseweb="tab-list"] > div { border-bottom: none !important; box-shadow: none !important; }
/* remove Streamlit's default underline / highlight / border line */
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"],
div[data-testid="stTabs"] hr {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* ---------- Cards / containers: white, floating, lift on hover ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border: 1px solid #dbe3ee !important;
    background: #ffffff !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 18px rgba(15, 23, 42, 0.06) !important;
    margin-bottom: 0.95rem !important;
    padding: 1.1rem 1.3rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.05), 0 14px 28px rgba(15, 23, 42, 0.10) !important;
    border-color: #c3d2e6 !important;
}
/* subject-scope card: white with a blue accent edge */
[data-testid="stVerticalBlockBorderWrapper"]:has(.unirag-scope-marker) {
    border-left: 5px solid #2563eb !important;
    background: linear-gradient(135deg, #ffffff 0%, #f6f9ff 100%) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.unirag-scope-marker) label p { font-weight: 700; color: #1e3a8a; }

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #dbe3ee !important;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 16px rgba(15, 23, 42, 0.06);
    margin-bottom: 0.9rem;
    transition: box-shadow 0.2s ease;
}
[data-testid="stChatMessage"]:hover { box-shadow: 0 2px 4px rgba(15, 23, 42, 0.05), 0 10px 22px rgba(15, 23, 42, 0.09); }

/* ---------- Buttons ---------- */
.stButton button, .stDownloadButton button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.90rem;
    transition: all 0.2s ease;
    border: 1px solid #d5dde8;
    background: #ffffff;
    color: #1e293b;
    padding: 7px 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: #93c5fd;
    background: #f8fbff;
    color: #1e3a8a;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.12);
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: 1px solid #1d4ed8 !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.30) !important;
}
.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(37, 99, 235, 0.35) !important;
}
/* disabled primary buttons must look disabled */
.stButton button[kind="primary"]:disabled,
.stButton button[kind="primary"][disabled],
.stFormSubmitButton button[kind="primary"]:disabled {
    background: #e2e8f0 !important;
    color: #64748b !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
}

/* ---------- Inputs, uploader, expanders ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {
    border-color: #cfd9e6 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-baseweb="select"] > div:hover, [data-baseweb="input"]:hover { border-color: #93c5fd !important; }
[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border: 2px dashed #93c5fd !important;
    border-radius: 12px !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"] section:hover { background: #f6f9ff !important; border-color: #2563eb !important; }
[data-testid="stExpander"] details {
    background: #ffffff;
    border: 1px solid #dbe3ee !important;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

/* keyed containers (st-key-<key> class is added by Streamlit itself) */
.st-key-welcome_card {
    background: linear-gradient(160deg, #eff6ff 0%, #e0e7ff 55%, #fae8ff 100%) !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 18px rgba(15, 23, 42, 0.08) !important;
}
.st-key-scope_card {
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 60%, #ffffff 100%) !important;
    border: 1px solid #93c5fd !important;
    border-left: 8px solid #2563eb !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 18px rgba(15, 23, 42, 0.06) !important;
}
.st-key-scope_card label p { font-weight: 800; color: #1e3a8a; font-size: 1rem; }
/* the dropdown itself: white, thick blue outline with a soft glow so it is the first thing you see */
.st-key-scope_card [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 2px solid #2563eb !important;
    border-radius: 10px !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.15), 0 4px 10px rgba(37, 99, 235, 0.15) !important;
    font-weight: 600;
}
.st-key-scope_card [data-baseweb="select"] > div:hover {
    box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.25), 0 6px 14px rgba(37, 99, 235, 0.22) !important;
}
/* welcome card: tint the whole container so the content is evenly spaced */
[data-testid="stVerticalBlockBorderWrapper"]:has(.unirag-welcome-marker) {
    background: linear-gradient(160deg, #eff6ff 0%, #e0e7ff 55%, #fae8ff 100%) !important;
}
/* ---------- Empty-state card ("pick a subject first") ---------- */
.unirag-empty {
    background: #ffffff; border: 1px dashed #93c5fd; border-radius: 14px;
    padding: 26px 24px; text-align: center; margin: 6px 0 10px 0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.unirag-empty-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 4px; }
.unirag-empty-text { font-size: 0.9rem; color: #475569; line-height: 1.55; }
.unirag-empty b { color: #1d4ed8; }

/* ---------- Section badges (unchanged styles) ---------- */
.unirag-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);   /* tiny lift so badges pop on the new background */
}
.unirag-badge-concept  { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
.unirag-badge-history  { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.unirag-badge-guidance { background: #f5f3ff; color: #5b21b6; border: 1px solid #ddd6fe; }
.unirag-badge-syllabus { background: #f0f9ff; color: #0369a1; border: 1px solid #bae6fd; }
.unirag-badge-trend    { background: #fff7ed; color: #c2410c; border: 1px solid #ffedd5; }
.unirag-badge-ranking  { background: #fefce8; color: #854d0e; border: 1px solid #fef08a; }
.unirag-badge-plain    { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

/* ---------- KPI summary cards ---------- */
.unirag-kpi-card {
    background: #ffffff;
    border: 1px solid #dbe3ee;
    border-top: 4px solid #2563eb;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 16px rgba(15, 23, 42, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.unirag-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.05), 0 14px 28px rgba(15, 23, 42, 0.10);
}
.unirag-kpi-card-label {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #475569;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.unirag-kpi-card-val { font-size: 1.65rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
.unirag-kpi-card-sub { font-size: 0.78rem; color: #2563eb; font-weight: 600; margin-top: 4px; }

/* ---------- Auth page ---------- */
.unirag-auth-header { text-align: center; margin-bottom: 20px; }
.unirag-auth-logo-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
    border-radius: 16px;
    color: white;
    font-size: 30px;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    margin-bottom: 12px;
}
.unirag-auth-card-title { font-size: 1.55rem; font-weight: 800; color: #0f172a; margin: 0 0 4px 0; letter-spacing: -0.02em; }
.unirag-auth-card-subtitle { font-size: 0.90rem; color: #475569; margin: 0; font-weight: 400; }
.unirag-auth-footer-badges {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
    font-size: 0.80rem;
    color: #475569;
    font-weight: 600;
}

@media (max-width: 768px) {
    .block-container { padding: 1.5rem 0.75rem 3rem 0.75rem !important; }
    .unirag-hero-banner { padding: 18px 16px !important; }
    .unirag-hero-title { font-size: 1.5rem !important; }
    [data-baseweb="tab"] { padding: 8px 14px !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STRUCTURED ANSWER RENDERING
# The backend returns ONE markdown string with section headers inside it
# (concept, exam history, guidance, syllabus, trend, ranking). We split it on
# those headers purely for display — nothing about the string is modified
# except dropping the "---" separator lines the backend put between sections.
# ---------------------------------------------------------------------------
def _classify_section_line(line):
    """Returns a section kind if this line STARTS a new section, else None.
    Only header-shaped lines count — e.g. the "💡 Simple way to understand
    it" line inside a concept explanation must NOT be mistaken for the
    "💡 STUDY & REVISION GUIDANCE" header."""
    s = line.strip()
    if not s:
        return None
    is_hash = s.startswith("#")
    if is_hash and "🏆" in s:
        return "ranking"
    if is_hash and "📈" in s:
        return "trend"
    if is_hash and "📊" in s:
        return "history"
    if s.startswith("💡 STUDY") or (is_hash and "STUDY & REVISION GUIDANCE" in s.upper()):
        return "guidance"
    if s.startswith("📋 SYLLABUS") or (is_hash and ("📋" in s or "SYLLABUS & NOTES RELEVANCE" in s.upper())):
        return "syllabus"
    if s.startswith("📘") or re.match(r"^##\s+[^#]", s):
        return "concept"
    return None


def split_answer_sections(answer):
    """Splits the backend answer string into [(kind, text), ...].
    Returns a single ("plain", answer) entry when no section marker exists."""
    if not answer:
        return [("plain", "")]
    sections = []
    current_kind, current_lines = None, []
    for line in answer.split("\n"):
        if line.strip() == "---":
            continue
        kind = _classify_section_line(line)
        # A 📘 line right after a "## TOPIC" heading (or any concept line while
        # already inside a concept section) continues the same section.
        if kind is not None and not (kind == "concept" and current_kind == "concept"):
            if current_lines and "".join(current_lines).strip():
                sections.append((current_kind or "plain", "\n".join(current_lines).strip()))
            current_kind, current_lines = kind, [line]
        else:
            current_lines.append(line)
    if current_lines and "".join(current_lines).strip():
        sections.append((current_kind or "plain", "\n".join(current_lines).strip()))
    if not sections or all(k == "plain" for k, _ in sections):
        return [("plain", answer)]
    return sections


_SECTION_CONFIG = {
    "concept": {
        "title": "Academic Concept & Grounded Theory",
        "icon": "📘",
    },
    "history": {
        "title": "University Examination History & Evidence",
        "icon": "📊",
    },
    "guidance": {
        "title": "Study Strategy & Exam Preparation Guidance",
        "icon": "💡",
    },
    "syllabus": {
        "title": "Official University Syllabus Alignment",
        "icon": "📋",
    },
    "trend": {
        "title": "Year-on-Year Exam Appearance Trends",
        "icon": "📈",
    },
    "ranking": {
        "title": "Most Repeated Examination Topics",
        "icon": "🏆",
    },
    "plain": {
        "title": "Academic Assistant Response",
        "icon": "🎓",
    },
}


def _clean_section_heading(text: str, kind: str) -> str:
    """Strips duplicate raw markdown headers that are already represented by the card badge."""
    lines = text.split("\n")
    if not lines:
        return text
    first = lines[0].strip()
    if kind == "history" and ("EXAM HISTORY" in first.upper() or "📊" in first):
        return "\n".join(lines[1:]).strip()
    if kind == "guidance" and ("STUDY & REVISION GUIDANCE" in first.upper() or "STUDY GUIDANCE" in first.upper() or "💡" in first):
        return "\n".join(lines[1:]).strip()
    if kind == "syllabus" and ("SYLLABUS" in first.upper() or "📋" in first):
        return "\n".join(lines[1:]).strip()
    if kind == "trend" and ("TREND" in first.upper() or "📈" in first):
        return "\n".join(lines[1:]).strip()
    if kind == "ranking" and ("REPEATED" in first.upper() or "🏆" in first):
        return "\n".join(lines[1:]).strip()
    return text


def render_structured_answer(answer):
    """Each detected section gets its own sleek bordered card with an icon badge."""
    for kind, text in split_answer_sections(answer):
        cfg = _SECTION_CONFIG.get(kind, _SECTION_CONFIG["plain"])
        cleaned = _clean_section_heading(text, kind)
        with st.container(border=True):
            st.markdown(
                f"<div style='margin-bottom: 8px;'><span class='unirag-badge unirag-badge-{kind}'>{cfg['icon']} {cfg['title']}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(cleaned)


def render_sources(sources):
    """Deduped-by-filename bulleted list (the backend often repeats the same
    PDF once per matched question)."""
    if not sources:
        return
    seen, unique = set(), []
    for s in sources:
        name = s.get("source", "Unknown") if isinstance(s, dict) else str(s)
        if name not in seen:
            seen.add(name)
            unique.append(name)
    with st.expander(f"📖 Sources ({len(unique)})"):
        for name in unique:
            st.markdown(f"- {name}")


def render_debug_info(debug_info):
    """'Behind the Answer' panel — only ever shown when Developer Mode is on.
    The debug_info data itself is always kept; this just gates the display."""
    if not debug_info or not st.session_state.get("dev_mode", False):
        return
    with st.expander("🔍 Behind the Answer & Query Intelligence"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**Canonical Topic:** `{debug_info.get('canonical_topic', debug_info.get('topic', '—'))}`")
            subtopics = debug_info.get("subtopics")
            if subtopics:
                st.markdown(f"**Subtopics:** {', '.join(f'`{s}`' for s in subtopics)}")
            intents = debug_info.get("intents") or ([debug_info.get("intent")] if debug_info.get("intent") else [])
            if intents:
                st.markdown(f"**Intents Detected:** {', '.join(f'`{i}`' for i in intents)}")
        with col_d2:
            st.markdown(f"**Exam Matching Questions:** `{debug_info.get('matching_questions', 0)}`")
            st.markdown(f"**Distinct Exam Papers:** `{debug_info.get('distinct_papers', 0)}`")
            years = debug_info.get("distinct_years")
            if years:
                st.markdown(f"**Exam Years:** `{', '.join(years)}`")
            if debug_info.get("is_followup"):
                st.markdown(f"**Follow-Up Reference:** Resolved to `{debug_info.get('resolved_reference')}`")

        sqs = debug_info.get("search_queries")
        if sqs:
            st.caption(f"🔍 Retrieval Queries: {', '.join(sqs)}")
        best_distance = debug_info.get("best_distance")
        threshold = debug_info.get("relevance_threshold")
        if best_distance is not None:
            verdict = "✅ below threshold (used)" if threshold is not None and best_distance <= threshold else "⚠️ above threshold (rejected)"
            st.caption(f"📏 Best Match Distance: `{best_distance}` (threshold `{threshold}`) — {verdict}")
        st.caption(f"⚙️ LLM Calls: {debug_info.get('llm_calls', 0)} • Pipeline: {debug_info.get('method', '—')}")



# ROLE GATE — a shared admin password gates document management/setup;
# everything else stays open. NOT per-student accounts — there's no
# database, no sessions beyond this browser tab's session_state. This
# runs BEFORE RAGPipeline loads, so the login screen appears instantly
# instead of waiting on FAISS/Ollama initialization first.
if "role" not in st.session_state:
    st.session_state.role = None
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "gate"  # "gate" (login) or "register" (its own page)

if st.session_state.role is None:
    st.markdown("""<style>
    .block-container { padding-top: 1.2rem !important; }
    .unirag-project-pill { display: none !important; }
    .unirag-auth-header { margin-bottom: 10px !important; }
    .unirag-auth-logo-badge { display: none !important; }  /* LAYOUT_PATCH: banner already carries the branding */
    </style>""", unsafe_allow_html=True)
    _, auth_col, _ = st.columns([1, 1.7, 1])   # LAYOUT_PATCH: banner + card share one column width
    with auth_col:
        st.markdown(f"""
        <div class="unirag-hero-banner" style="text-align: center; padding: 14px 18px; margin-bottom: 10px;">
            <div class="unirag-project-pill">🎓 MCA FINAL SEMESTER PROJECT · SECURE ACADEMIC PORTAL</div>
            <div class="unirag-hero-title" style="font-size: 1.7rem; font-weight: 800; margin-bottom: 4px;">uniRAG <span>Academic Intelligence</span></div>
            <p class="unirag-hero-subtitle" style="margin-bottom: 0; font-size: 0.84rem; white-space: nowrap;">Grounded in <b>{UNIVERSITY_NAME}</b> · <b>{DEPARTMENT}</b> Notes, Question Papers & Syllabus</p>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            if st.session_state.auth_view == "register":
                st.markdown("""
                <div class="unirag-auth-header">
                    <div class="unirag-auth-logo-badge">📝</div>
                    <div class="unirag-auth-card-title">Student Registration</div>
                    <div class="unirag-auth-card-subtitle">Create your personal offline study account</div>
                </div>
                """, unsafe_allow_html=True)
                with st.form("self_register_form"):
                    reg_name = st.text_input("Full Name", placeholder="e.g. Raghav Sharma")
                    reg_email = st.text_input("University / Student Email", placeholder="student@university.edu")
                    reg_password = st.text_input("Password", type="password", placeholder="Enter secure password")
                    reg_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("✨ Create Account", type="primary", use_container_width=True)
                    if submitted:
                        existing = _load_students()
                        normalized_email = reg_email.lower().strip()
                        if not reg_name or not reg_email or not reg_password:
                            st.error("Name, email, and password are all required.")
                        elif reg_password != reg_password_confirm:
                            st.error("Passwords don't match.")
                        elif normalized_email in existing:
                            st.error("An account with this email already exists — log in instead.")
                        else:
                            try:
                                _register_student(reg_email, reg_name, reg_password)
                                st.success("Account created successfully! Click below to sign in.")
                            except StudentFileUnreadable:
                                st.error("The accounts file could not be read, so registration is paused to "
                                         "protect existing accounts. Ask the administrator to check students.json.")

                st.markdown("<div style='text-align:center; margin-top:12px;'>", unsafe_allow_html=True)
                if st.button("← Already registered? Sign In", use_container_width=True):
                    st.session_state.auth_view = "gate"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="unirag-auth-header">
                    <div class="unirag-auth-logo-badge">🎓</div>
                    <div class="unirag-auth-card-title">Sign In to uniRAG</div>
                    <div class="unirag-auth-card-subtitle">Select your access role to continue</div>
                </div>
                """, unsafe_allow_html=True)

                choice = st.radio("Access Level", ["👨‍🎓 Student", "🛡️ Faculty / Admin"], horizontal=True, label_visibility="collapsed")
                st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

                if "Admin" in choice:
                    with st.form("admin_login_form", border=False):  # LOGIN_SESSION_PATCH
                        st.markdown("<div style='font-size:0.86rem; font-weight:600; color:#334155; margin-bottom:4px;'>Administrator Passcode</div>", unsafe_allow_html=True)
                        pw = st.text_input("Admin Password", type="password", placeholder="Enter faculty passcode", label_visibility="collapsed")
                        st.caption("🔒 Department administrator privileges for corpus indexing, paper uploads & diagnostics.")
                        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                        if st.form_submit_button("🚀 Sign In as Administrator", type="primary", use_container_width=True):
                            _left = _login_locked("admin")
                            if _left:
                                st.error(f"Too many failed attempts. Try again in {_left} s.")
                            elif hmac.compare_digest(pw.encode("utf-8"), str(ADMIN_PASSWORD).encode("utf-8")):
                                _login_reset("admin")
                                _reset_user_session()
                                st.session_state.role = "admin"
                                st.rerun()
                            else:
                                _login_failed("admin")
                                st.error("Incorrect password.")
                else:
                    with st.form("student_login_form", border=False):  # LOGIN_SESSION_PATCH
                        st.markdown("<div style='font-size:0.86rem; font-weight:600; color:#334155; margin-bottom:4px;'>Student Email</div>", unsafe_allow_html=True)
                        student_email = st.text_input("Student Email", placeholder="student@university.edu", label_visibility="collapsed")
                        st.markdown("<div style='font-size:0.86rem; font-weight:600; color:#334155; margin-top:8px; margin-bottom:4px;'>Password</div>", unsafe_allow_html=True)
                        student_password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
                        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                        if st.form_submit_button("🚀 Sign In to Workspace", type="primary", use_container_width=True):
                            if not student_email or not student_password:
                                st.error("Enter both email and password.")
                            elif _login_locked(student_email.lower().strip()):
                                st.error(f"Account locked due to failed attempts ({_login_locked(student_email.lower().strip())}s remaining).")
                            elif _verify_student(student_email, student_password):
                                _login_reset(student_email.lower().strip())
                                _email_ok = student_email.lower().strip()
                                _reset_user_session()
                                st.session_state.role = "student"
                                st.session_state.student_email = _email_ok
                                st.session_state.student_name = _load_students().get(st.session_state.student_email, {}).get("name", "")
                                st.rerun()
                            else:
                                _login_failed(student_email.lower().strip())
                                st.error("Invalid email or password.")

                    st.markdown("<div style='text-align:center; margin:16px 0 6px 0; border-top:1px solid #f1f5f9; padding-top:14px; font-size:0.85rem; color:#475569;'>New student?</div>", unsafe_allow_html=True)
                    if st.button("📝 Register New Student Account", use_container_width=True):
                        st.session_state.auth_view = "register"
                        st.rerun()

        st.markdown(f"""
        <div class="unirag-auth-footer-badges">
            <span>🔒 100% Offline Local Inference</span>
            <span>•</span>
            <span>⚡ Mistral 7B + nomic-embed-text</span>
            <span>•</span>
            <span>🏛️ {UNIVERSITY_NAME}</span>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# INITIALIZE RAG
@st.cache_resource
def get_rag():
    return RAGPipeline()

rag = get_rag()

with st.sidebar:
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.08); border-radius: 14px; padding: 14px 16px; border: 1px solid rgba(255,255,255,0.18); box-shadow: 0 6px 18px rgba(0,0,0,0.25); margin-bottom: 14px;'>
        <div style='display: flex; align-items: center; gap: 12px;'>
            <div style='display: flex; align-items: center; justify-content: center; width: 44px; height: 44px; background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%); border-radius: 10px; color: white; font-size: 22px; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);'>🎓</div>
            <div>
                <div style='font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;'>uni<span style="color:#60a5fa">RAG</span></div>
                <div style='font-size: 0.76rem; color: #cbd5e1; font-weight: 500;'>{UNIVERSITY_NAME} · {DEPARTMENT}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    role_label = html.escape(st.session_state.get("student_name") or (st.session_state.role.title() if st.session_state.role else "Guest"))
    role_icon = "👨‍🎓" if st.session_state.role == "student" else "🛡️"
    st.markdown(f"""
    <div style='padding: 9px 12px; background: rgba(37,99,235,0.35); border: 1px solid rgba(147,197,253,0.45); border-radius: 10px; font-size: 0.82rem; font-weight: 600; color: #ffffff; margin-bottom: 8px;'>
        {role_icon} Logged in as: <b>{role_label}</b>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔓 Switch role", use_container_width=True):
        _reset_user_session()
        st.session_state.role = None
        st.rerun()

    if st.button("🧹 New chat", use_container_width=True, help="Clears the conversation and forgets the last topic."):
        st.session_state.messages = []
        st.session_state.pop("last_topic", None)
        st.rerun()

    st.divider()
    st.toggle(
        "🔧 Developer Mode",
        value=False,
        key="dev_mode",
        help="Shows the 'Behind the Answer & Query Intelligence' panel under each answer.",
    )

    st.markdown(f"""
    <div style='margin-top: 20px; padding: 14px; background: rgba(255,255,255,0.07); border-radius: 14px; border: 1px solid rgba(255,255,255,0.16); font-size: 0.76rem; color: #cbd5e1; box-shadow: 0 6px 18px rgba(0,0,0,0.2);'>
        <div style='font-weight: 700; color: #ffffff; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; letter-spacing: 0.04em;'>⚡ uniRAG ENGINE STATUS</div>
        <div style='margin-bottom: 3px;'>🟢 <b style='color:#ffffff'>Model:</b> Mistral 7B (Ollama)</div>
        <div style='margin-bottom: 3px;'>🟢 <b style='color:#ffffff'>Embeddings:</b> nomic-embed-text</div>
        <div style='margin-bottom: 3px;'>🟢 <b style='color:#ffffff'>Question Bank:</b> {len(rag._question_bank):,} indexed</div>
        <div>🟢 <b style='color:#ffffff'>Inference:</b> 100% Offline (Local)</div>
    </div>
    """, unsafe_allow_html=True)

# HEADER & BRANDING  (LAYOUT_PATCH: compact single-row banner)
st.markdown(f"""
<div class="unirag-hero-banner unirag-hero-compact">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div class="unirag-crest-badge">🎓</div>
            <div>
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:6px;">
                    <div class="unirag-hero-title">uniRAG <span>Academic Intelligence</span></div>
                    <div class="unirag-project-pill">MCA Final Semester Project</div>
                </div>
                <div class="unirag-hero-subtitle">Grounded in <b>{UNIVERSITY_NAME}</b> · <b>{DEPARTMENT}</b> Notes, Question Papers & Syllabus</div>
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="unirag-hero-chip">📚 <b>{len(rag.get_available_subjects())}</b> Subjects incl. Labs</span>
            <span class="unirag-hero-chip">📄 <b>{len(rag._question_bank):,}</b> Exam Questions</span>
            <span class="unirag-hero-chip">🔒 100% Offline AI</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ACADEMIC COURSE / SUBJECT SELECTOR  (kept in the main area; COLOR_PATCH: styled as a tinted card)
with st.container(border=True, key="scope_card"):
    st.markdown("<div class='unirag-scope-marker'></div>", unsafe_allow_html=True)
    sub_col1, sub_col2 = st.columns([0.62, 0.38])
    with sub_col1:
        available_subjects = rag.get_available_subjects()
        subject_options = ["All / Auto-detect"] + available_subjects
        selected_subject = st.selectbox(
            "📚 Academic Course / Subject Scope:",
            subject_options,
            index=0,
            help="Hard-scopes FAISS retrieval to this subject index, or keep 'All / Auto-detect' for cross-subject matching."
        )
    with sub_col2:
        st.write("")
        st.caption("💡 *Auto-detect intelligently matches relevant notes, syllabus, and past exam questions.*")

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_question(question, marks, subject=None):
    """Streams the definition answer token-by-token for a snappier feel on
    CPU-only inference; frequency/importance/trend/ranking answers are
    already instant (no LLM call for the deterministic ones) so they
    render in one shot."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        last_topic = st.session_state.get("last_topic")
        placeholder = st.empty()
        sources = []
        topic = None
        full_answer = ""
        debug_info = None
        start_time = time.time()

        try:
            with st.spinner("🔍 Thinking..."):
                gen = rag.ask_stream(question, marks, subject=subject, context_topic=last_topic)
                for kind, payload in gen:
                    if kind == "meta":
                        sources = payload.get("sources", [])
                        topic = payload.get("topic")
                    elif kind == "token":
                        full_answer = payload
                        placeholder.markdown(full_answer + "▌")
                    elif kind == "done":
                        full_answer = payload["answer"]
                        sources = payload.get("sources", sources)
                        topic = payload.get("topic", topic)
                        debug_info = payload.get("debug_info")
        except Exception as e:
            # A raw traceback here would be a bad look mid-demo, and it can
            # happen for a purely environmental reason (Ollama not running,
            # model not pulled yet) that has nothing to do with the question.
            placeholder.empty()
            full_answer = (
                "⚠️ Something went wrong answering that — this usually means Ollama isn't "
                "running or the model isn't loaded. Check that `ollama serve` is up, then try again."
            )
            st.error(full_answer)
            with st.expander("Technical details"):
                st.code(f"{type(e).__name__}: {e}")
            st.session_state.messages.append({"role": "assistant", "content": full_answer, "sources": [], "debug_info": None})
            return

        elapsed = time.time() - start_time

        # Streaming is done: swap the raw streamed text for the sectioned view.
        placeholder.empty()
        render_structured_answer(full_answer)
        if topic:
            st.session_state["last_topic"] = topic

        subject_used = (debug_info or {}).get("subject") or subject
        badge_bits = []
        if subject_used and subject_used != "All / Auto-detect":
            badge_bits.append(f"📚 {subject_used}")
        badge_bits.append(f"⏱️ {elapsed:.0f}s")
        st.caption(" · ".join(badge_bits))

        render_sources(sources)
        render_debug_info(debug_info)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
        "debug_info": debug_info,
    })


def extract_questions_from_pdf(pdf_path):
    """Uses the same per-page OCR fallback as the main ingestion pipeline
    — a scanned question paper returns blank text from PyPDFLoader alone.
    Actual extraction is shared with document_loader.extract_questions_from_text.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    page_texts = []
    for i, doc in enumerate(docs):
        text = doc.page_content
        if len(text.strip()) < MIN_TEXT_LENGTH:
            text = ocr_page(pdf_path, i, psm=4)
        page_texts.append(text)

    questions = extract_questions_from_text(page_texts)

    return questions


def generate_answers_file(questions, results=None, subject=None):
    """Builds the downloadable text file. When the answers were already
    computed for on-screen display (results), they are reused instead of
    asking every question a second time — output is identical, just half
    the wait on CPU-only hardware."""
    content = "=" * 50 + "\n"
    content += "QUESTION PAPER ANSWERS\n"
    content += "=" * 50 + "\n\n"

    for i, q in enumerate(questions, 1):
        if results is not None and i - 1 < len(results):
            result = results[i - 1]
        else:
            try:
                result = rag.ask(q['text'], marks=q['marks'], subject=subject, plain_answer_only=True)
            except Exception as e:
                result = {"answer": f"[This question failed to answer: {type(e).__name__}: {e}]"}
        content += f"Q{i}: {q['text']}\n"
        content += f"Marks: {q['marks']}\n"
        content += "-" * 40 + "\n"
        content += f"Answer: {result.get('answer', '')}\n\n"

    return content


def generate_repeated_answers_text(items, subject=None):
    """Builds a formatted text file containing repeated exam questions and their answers."""
    lines = [
        "=" * 60,
        "uniRAG ACADEMIC INTELLIGENCE · MOST REPEATED EXAM QUESTIONS",
        f"Subject Scope: {subject or 'All Subjects'}",
        f"Institution: {UNIVERSITY_NAME} · {DEPARTMENT} Department",
        f"Generated: {time.strftime('%d %B %Y')}",
        "=" * 60,
        "",
    ]
    for it in items:
        idx = it["index"]
        q = it["question"]
        marks = it.get("marks") or DEFAULT_CHAT_MARKS
        years = ", ".join(it.get("years", [])) or "N/A"
        sittings = it.get("sittings", len(it.get("years", [])))
        ans = _rs_clean(it.get("answer"), pdf=False) or "[Answer not generated yet]"

        lines.append(f"QUESTION #{idx} ({marks} Marks)")
        lines.append(f"Historical Frequency: Tested across {sittings} Exam Sittings [{years}]")
        lines.append(f"Question: {q}")
        lines.append("-" * 50)
        lines.append("Model Answer (Grounded in University Notes & Syllabus):")
        lines.append(ans)
        lines.append("")
        lines.append("=" * 60)
        lines.append("")
    return "\n".join(lines)


def generate_repeated_answers_pdf(items, subject=None):
    """Generates a structured, professional academic PDF booklet of repeated
    exam questions and model answers using PyMuPDF."""
    import pymupdf
    import textwrap

    doc = pymupdf.open()
    page_w = 595
    page_h = 842
    margin_l = 45
    margin_r = 550
    margin_b = 50
    line_h = 13.5
    wrap_width = 82

    current_page = None
    y = 50

    def add_page():
        nonlocal current_page, y
        current_page = doc.new_page(width=page_w, height=page_h)
        # Top banner line
        current_page.draw_rect(pymupdf.Rect(margin_l, 28, margin_r, 30), color=(0.14, 0.38, 0.92), fill=(0.14, 0.38, 0.92))
        current_page.insert_text(pymupdf.Point(margin_l, 24), f"uniRAG · {subject or 'All Subjects'} · Repeated Exam Questions & Answers", fontsize=8, color=(0.4, 0.45, 0.5))
        # Footer
        current_page.insert_text(pymupdf.Point(margin_l, page_h - 22), f"{UNIVERSITY_NAME} · {DEPARTMENT} | 100% Offline AI Local Inference", fontsize=8, color=(0.5, 0.5, 0.5))
        y = 50

    add_page()

    # Document Header Title
    current_page.insert_text(pymupdf.Point(margin_l, y), "uniRAG Academic Intelligence", fontsize=18, fontname="helv", color=(0.06, 0.09, 0.16))
    y += 20
    current_page.insert_text(pymupdf.Point(margin_l, y), "Most Repeated Examination Questions & Answer Key", fontsize=12, fontname="helv", color=(0.14, 0.38, 0.92))
    y += 16
    current_page.insert_text(pymupdf.Point(margin_l, y), f"Subject: {subject or 'All Subjects'} · Generated on {time.strftime('%d %B %Y')}", fontsize=9, fontname="helv", color=(0.4, 0.45, 0.5))
    y += 26

    for it in items:
        if y > page_h - 110:
            add_page()

        idx = it["index"]
        q = _rs_clean(it["question"], pdf=True)  # PDF_SYMBOLS_PATCH
        marks = it.get("marks") or DEFAULT_CHAT_MARKS
        years = ", ".join(it.get("years", [])) or "N/A"
        sittings = it.get("sittings", len(it.get("years", [])))
        ans = _rs_clean(it.get("answer"), pdf=True) or "[Answer not generated yet]"

        # Question header card background
        current_page.draw_rect(pymupdf.Rect(margin_l, y - 9, margin_r, y + 9), color=None, fill=(0.93, 0.95, 0.98))
        current_page.insert_text(pymupdf.Point(margin_l + 8, y + 3), f"Question #{idx} ({marks} Marks) · Tested in {sittings} Exam Sittings [{years}]", fontsize=10, fontname="helv", color=(0.1, 0.25, 0.7))
        y += 24

        # Question text
        for line in textwrap.wrap(q, width=wrap_width):
            if y > page_h - margin_b:
                add_page()
            current_page.insert_text(pymupdf.Point(margin_l, y), line, fontsize=10, fontname="helv", color=(0.06, 0.09, 0.16))
            y += line_h + 1
        y += 6

        # Answer label
        if y > page_h - margin_b:
            add_page()
        current_page.insert_text(pymupdf.Point(margin_l, y), "Model Answer:", fontsize=9.5, fontname="helv", color=(0.14, 0.38, 0.92))
        y += line_h

        # Answer paragraphs  (PDF_LAYOUT_PATCH: tables / diagrams keep their layout in a fixed-width font)
        import re as _re
        for p in ans.split("\n"):
            p_clean = p.strip()
            if not p_clean:
                y += 5
                continue
            p_raw = p.rstrip().expandtabs(4)
            _indent = len(p_raw) - len(p_raw.lstrip(" "))
            _is_pre = (
                (_indent >= 2 and not p_clean.startswith("- ") and len(p_clean) <= 40)
                or (p_raw.count(" | ") >= 2 and "=" not in p_raw)
                or len(_re.findall(r"\S {2,}\S", p_raw)) >= 2
                or len(_re.findall(r"\S {3,}\S", p_raw)) >= 1
            )
            if _is_pre:
                for line in (textwrap.wrap(p_raw, width=100, subsequent_indent=" " * (_indent + 2)) or [""]):
                    if y > page_h - margin_b:
                        add_page()
                    current_page.insert_text(pymupdf.Point(margin_l, y), line, fontsize=8, fontname="cour", color=(0.15, 0.2, 0.25))
                    y += 11.5
                y += 1
                continue
            for line in textwrap.wrap(p_clean, width=wrap_width):
                if y > page_h - margin_b:
                    add_page()
                current_page.insert_text(pymupdf.Point(margin_l, y), line, fontsize=9.2, fontname="helv", color=(0.15, 0.2, 0.25))
                y += line_h
            y += 2
        y += 18

    return doc.write()


# Primary Academic Navigation
_is_admin = st.session_state.role == "admin"
_nav_tabs = [
    "💬 Study Assistant",
    "📝 Question Paper Solver",
    "📈 Exam Intelligence",
    "📚 Resource Library",
] + (["🛠️ System Setup"] if _is_admin else [])

_tabs = st.tabs(_nav_tabs)
tab_ask = _tabs[0]
tab_solver = _tabs[1]
tab_intel = _tabs[2]
tab_library = _tabs[3]
tab_admin = _tabs[4] if _is_admin else None

# TAB 1: STUDY ASSISTANT
with tab_ask:
    if not st.session_state.messages:
        with st.container(border=True, key="welcome_card"):
            st.markdown(f"""
            <div class='unirag-welcome-marker' style='text-align: center; padding: 10px 18px 34px 18px;'>
                <div style='display: inline-flex; align-items: center; justify-content: center; width: 56px; height: 56px; background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%); border-radius: 16px; color: white; font-size: 28px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25); margin-bottom: 8px;'>🎓</div>
                <div style='font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;'>
                    👋 Welcome to uniRAG
                </div>
                <div style='font-size: 0.95rem; font-weight: 600; color: #2563eb; margin-bottom: 10px;'>
                    {UNIVERSITY_NAME} · {DEPARTMENT} Department Intelligent Academic Assistant
                </div>
                <div style='font-size: 0.92rem; color: #475569; max-width: 640px; margin: 0 auto 18px auto; line-height: 1.6;'>
                    Ask conceptual questions, explore previous-year exam frequencies, or upload question papers. 
                    Every answer is calibrated to printed marks weightage and grounded strictly in official university lecture notes, curriculum syllabus, and past exam papers.
                </div>
                <div style='display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;'>
                    <span class='unirag-badge unirag-badge-concept'>📘 Verified Notes</span>
                    <span class='unirag-badge unirag-badge-history'>📊 {len(rag._question_bank):,} Exam Questions</span>
                    <span class='unirag-badge unirag-badge-guidance'>💡 Revision Guidance</span>
                    <span class='unirag-badge unirag-badge-syllabus'>📋 Syllabus Aligned</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Replay conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_structured_answer(msg["content"])
                render_sources(msg.get("sources"))
                render_debug_info(msg.get("debug_info"))
            else:
                st.write(msg["content"])

    # Chat Input bar (text only — image attachments removed)
    chat_text = st.chat_input("Type your question...")

    if chat_text:
        question = chat_text.strip()
        if question:
            process_question(question, DEFAULT_CHAT_MARKS, selected_subject)
            st.rerun()
        else:
            st.error("Please type a question.")

# TAB 2: QUESTION PAPER SOLVER (AI-answers from uploaded question paper PDF)
with tab_solver:
    st.caption("Upload an examination question paper PDF to automatically extract questions and generate calibrated answers according to each question's printed marks.")

    uploaded_pdf = st.file_uploader(
        "Upload Question Paper (PDF)",
        type=['pdf'],
        key="question_paper_uploader"
    )

    pdf_path_to_process = None
    pdf_display_name = None
    is_temp_file = False

    if uploaded_pdf:
        pdf_display_name = uploaded_pdf.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_pdf.getbuffer())
            pdf_path_to_process = tmp.name
            is_temp_file = True

    if pdf_path_to_process and pdf_display_name:
        st.markdown(f"**Uploaded Paper:** `{pdf_display_name}`")
        if st.button("🔍 Extract Questions & Generate Calibrated Answers", type="primary", key="btn_solve_paper"):
            with st.spinner("📖 Extracting questions from question paper..."):
                questions = extract_questions_from_pdf(pdf_path_to_process)

                if questions:
                    st.success(f"✅ Found {len(questions)} questions")
                    st.caption("⏳ Each question is answered with depth matching its printed marks. Progress below.")
                    total_q = len(questions)
                    progress_bar = st.progress(0.0)
                    progress_label = st.empty()
                    collected_results = []
                    failed = []

                    for i, q in enumerate(questions, 1):
                        preview = q['text'][:60] + ("..." if len(q['text']) > 60 else "")
                        progress_label.text(f"⏳ Answering Q{i}/{total_q}: {preview}")
                        st.markdown(f"### Q{i}: {q['text']}")
                        st.caption(f"📝 {q['marks']} marks")

                        try:
                            result = rag.ask(q['text'], marks=q['marks'], subject=selected_subject, plain_answer_only=True)
                        except Exception as e:
                            result = {"answer": "", "sources": [], "debug_info": None}
                            st.error(f"⚠️ This question failed to answer: {type(e).__name__}: {e}")
                            failed.append({"n": i, "text": q["text"], "error": str(e)})

                        collected_results.append(result)
                        if not result or not result.get("answer"):
                            st.warning("No answer was produced for this question.")
                        else:
                            st.write(result["answer"])
                            render_sources(result.get("sources"))

                        st.divider()
                        progress_bar.progress(i / total_q)

                    if failed:
                        st.session_state["qp_last_failed"] = failed
                        st.warning(f"⚠️ {len(failed)} of {total_q} question(s) failed to answer.")

                    progress_label.text(f"✅ Completed — answered all {total_q} questions.")
                    st.session_state["qp_saved"] = {
                        "file": pdf_display_name, "questions": questions, "results": collected_results,
                    }

                    st.download_button(
                        "📥 Download All Answers",
                        generate_answers_file(questions, results=collected_results),
                        file_name=f"{Path(pdf_display_name).stem}_answers.txt",
                        mime="text/plain",
                        key="btn_download_solved_answers"
                    )
                else:
                    st.error("❌ No questions found in the PDF.")

                if is_temp_file:
                    try:
                        os.unlink(pdf_path_to_process)
                    except OSError:
                        pass

        elif st.session_state.get("qp_saved", {}).get("file") == pdf_display_name:
            _qp = st.session_state["qp_saved"]
            for _i, (_q, _res) in enumerate(zip(_qp["questions"], _qp["results"]), 1):
                st.markdown(f"### Q{_i}: {_q['text']}")
                st.caption(f"📝 {_q['marks']} marks")
                if not _res or not _res.get("answer"):
                    st.warning("No answer was produced for this question.")
                else:
                    st.write(_res["answer"])
                    render_sources(_res.get("sources"))
                st.divider()
            st.download_button(
                "📥 Download All Answers",
                generate_answers_file(_qp["questions"], results=_qp["results"]),
                file_name=f"{Path(pdf_display_name).stem}_answers.txt",
                mime="text/plain",
                key="qp_download_saved_btn",
            )

# TAB 3: EXAM INTELLIGENCE & TRENDS
with tab_intel:
    st.caption("Analyze examination patterns, topic frequency, repeated questions, and question paper trends.")

    distinct_years = sorted(list({r.year for r in rag._question_bank if r.year}))
    distinct_subjects = len({r.subject for r in rag._question_bank if r.subject})
    long_answer_count = sum(1 for r in rag._question_bank if r.marks and r.marks >= 8)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="unirag-kpi-card" style="border-top-color:#2563eb; background:linear-gradient(180deg,#eff6ff 0%,#ffffff 100%);">
            <div class="unirag-kpi-card-label">Total Exam Questions</div>
            <div class="unirag-kpi-card-val">{len(rag._question_bank):,}</div>
            <div class="unirag-kpi-card-sub">📄 Deterministic OCR Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        years_span = f"{distinct_years[0]} – {distinct_years[-1]}" if distinct_years else "N/A"
        st.markdown(f"""
        <div class="unirag-kpi-card" style="border-top-color:#16a34a; background:linear-gradient(180deg,#f0fdf4 0%,#ffffff 100%);">
            <div class="unirag-kpi-card-label">Exam Sittings Covered</div>
            <div class="unirag-kpi-card-val">{len(distinct_years)} Years</div>
            <div class="unirag-kpi-card-sub">📅 {years_span}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="unirag-kpi-card" style="border-top-color:#7c3aed; background:linear-gradient(180deg,#f5f3ff 0%,#ffffff 100%);">
            <div class="unirag-kpi-card-label">Subjects with Papers</div>
            <div class="unirag-kpi-card-val">{distinct_subjects}</div>
            <div class="unirag-kpi-card-sub">🏛️ Bangalore University MCA</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="unirag-kpi-card" style="border-top-color:#ea580c; background:linear-gradient(180deg,#fff7ed 0%,#ffffff 100%);">
            <div class="unirag-kpi-card-label">High-Weightage (8+ Marks)</div>
            <div class="unirag-kpi-card-val">{long_answer_count:,}</div>
            <div class="unirag-kpi-card-sub">⭐ Core Descriptive & Problems</div>
        </div>
        """, unsafe_allow_html=True)

    if selected_subject and _LAB_RE.search(selected_subject):
        st.info(
            f"🧪 **{selected_subject}** is a practical laboratory course assessed via lab practical exams and viva voce. "
            "Exam Intelligence analyzes semester-end university theory examination papers. "
            "Please select a theory subject (e.g. DS, DAA, OS, DBMS, TOC) or 'All / Auto-detect' to explore question paper intelligence."
        )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    intel_sub1, intel_sub2, intel_sub3 = st.tabs([
        "📈 Topic Year-by-Year Trend",
        "🏆 Most Repeated Questions",
        "🎯 High-Weightage Questions by Marks",
    ])

    with intel_sub1:
        _t_all = (not selected_subject) or selected_subject == "All / Auto-detect"  # REQUIRE_SUBJECT_TABS_PATCH
        if _t_all:
            st.markdown(
                """<div class="unirag-empty"><div class="unirag-empty-title">📌 Pick a subject to check a topic's trend</div>
                <div class="unirag-empty-text">Choose a subject in the <b>Academic Course / Subject Scope</b> dropdown just above the tabs to see how a topic has trended across its exam years.</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#334155; margin-bottom:6px;'>⚡ Quick Trend Topics (Click to inspect):</div>", unsafe_allow_html=True)
            q_cols = st.columns(6)
            preset_topics = ["Queue", "Quick Sort", "Decision Tree", "Normalization", "Deadlock", "Binary Tree"]
            clicked_preset = None
            for idx, p_topic in enumerate(preset_topics):
                with q_cols[idx]:
                    if st.button(p_topic, key=f"trend_chip_{p_topic}", use_container_width=True, disabled=_t_all):
                        st.session_state["selected_trend_topic"] = p_topic
                        clicked_preset = p_topic

            active_trend_topic = st.session_state.get("selected_trend_topic", "")
            trend_topic = st.text_input(
                "Topic to check across past exam years:",
                value=active_trend_topic,
                placeholder="e.g. queue, normalization, deadlock, quick sort...",
                disabled=_t_all
            )
            run_trend = (st.button("📈 Show Trend", key="btn_show_trend", type="primary", disabled=_t_all) or (clicked_preset is not None)) and not _t_all
            if run_trend and trend_topic:
                with st.spinner(f"Analyzing '{trend_topic}' across years..."):
                    result = rag.analyze_topic_trend_by_questions(trend_topic, selected_subject)
                if result is None:
                    st.error("Please type a more specific topic name (at least 3 characters).")
                elif not result["year_counts"]:
                    _scope_txt = f" in {selected_subject}" if selected_subject and selected_subject != "All / Auto-detect" else ""
                    msg = f"No year-tagged questions about '{trend_topic}' found{_scope_txt}. Try another topic or subject."  # TREND_FIX
                    if result["undated_count"]:
                        msg += f" ({result['undated_count']} matching question(s) found, but without a detectable year.)"
                    st.warning(msg)
                else:
                    st.markdown(f"**Topic:** `{result['topic']}` | **Trend:** **{result['trend'].upper()}**")
                    if len(result["year_counts"]) >= 2:
                        import altair as alt
                        trend_df = pd.DataFrame({
                            "Year": [str(y) for y in result["year_counts"].keys()],
                            "Questions": list(result["year_counts"].values()),
                        })
                        _x = alt.X("Year:O", title="Exam year", axis=alt.Axis(labelAngle=0))
                        _y = alt.Y("Questions:Q", title="Matching questions",
                                   axis=alt.Axis(tickMinStep=1, format="d"),
                                   scale=alt.Scale(domainMin=0))
                        bars = alt.Chart(trend_df).mark_bar(
                            color="#2563eb", opacity=0.85, cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=38
                        ).encode(x=_x, y=_y, tooltip=["Year", "Questions"])
                        labels = alt.Chart(trend_df).mark_text(
                            dy=-9, fontWeight="bold", color="#0f172a"
                        ).encode(x=_x, y=_y, text=alt.Text("Questions:Q", format="d"))
                        line = alt.Chart(trend_df).mark_line(
                            color="#f97316", strokeWidth=2.5, point=alt.OverlayMarkDef(color="#f97316", size=60)
                        ).encode(x=_x, y=_y)
                        st.altair_chart(
                            (bars + line + labels).properties(
                                height=320, title=f"'{result['topic']}' - questions per exam year ({result['trend'].upper()})"
                            ),
                            use_container_width=True,
                        )
                    else:
                        only_year, only_count = next(iter(result["year_counts"].items()))
                        st.write(f"{only_year}: {only_count} question(s) — need at least 2 years with a match to chart a trend.")
                    if result["undated_count"]:
                        st.caption(f"({result['undated_count']} additional matching question(s) had no detectable year, not shown above)")
                    st.caption("⚠️ Based on indexed university question papers (number of matching questions per year).")

    with intel_sub2:
        _all_scope = (not selected_subject) or selected_subject == "All / Auto-detect"  # REQUIRE_SUBJECT_PATCH
        if _all_scope:
            st.markdown(
                """<div class="unirag-empty"><div class="unirag-empty-title">📌 Pick a subject to see its most repeated questions</div>
                <div class="unirag-empty-text">Choose a subject in the <b>Academic Course / Subject Scope</b> dropdown just above the tabs to list the questions repeated across exam sittings and download its solutions bundle.</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"Analyze questions that have appeared repeatedly across multiple exam sittings in **{selected_subject}**.")
            if st.button("🔍 Find Most Repeated Questions", key="btn_find_repeated", type="primary", disabled=_all_scope):
                st.session_state["show_repeated_questions"] = True

            if st.session_state.get("show_repeated_questions", False) and not _all_scope:
                with st.spinner("Analyzing question bank for recurring exam themes..."):
                    repeated = rag.exam_intelligence.get_most_repeated_topics_from_bank(
                        rag._question_bank, subject=selected_subject, top_k=12
                    )
                if not repeated:
                    st.info(f"No recurring question clusters found for {selected_subject}. Ensure question papers are indexed.")
                else:
                    st.success(f"Found {len(repeated)} frequently repeated questions/themes across multiple exam sittings:")

                    # Initialize cache dictionary in session_state if not present
                    if "repeated_answers_cache" not in st.session_state:
                        st.session_state["repeated_answers_cache"] = {}

                    cached_answers = st.session_state["repeated_answers_cache"]
                    active_subj_key = (selected_subject or "All").strip().upper()

                    # Build prepared items list
                    prepared_items = []
                    for idx, r in enumerate(repeated, 1):
                        q_text = r['representative_question']
                        m_val = r.get('marks') or DEFAULT_CHAT_MARKS
                        ans_val = (cached_answers.get(f"{active_subj_key}::{q_text}")
                                   or _rs_lookup(r.get('subject') or selected_subject, q_text))
                        prepared_items.append({
                            "index": idx,
                            "question": q_text,
                            "marks": m_val,
                            "years": r.get('years', []),
                            "sittings": len(r.get('years', [])),
                            "subject": r.get('subject', selected_subject),
                            "answer": ans_val
                        })

                    answered_count = sum(1 for it in prepared_items if it["answer"])

                    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

                    for idx, r in enumerate(repeated, 1):
                        y_str = ", ".join(r['years'][:4]) if r['years'] else "N/A"
                        q_text = r['representative_question']
                        m_val = r.get('marks') or DEFAULT_CHAT_MARKS
                        cache_key = f"{active_subj_key}::{q_text}"
                        ans_text = cached_answers.get(cache_key) or _rs_lookup(r.get('subject') or selected_subject, q_text)

                        with st.container(border=True):
                            c_num, c_body = st.columns([0.07, 0.93])
                            with c_num:
                                st.markdown(f"<div style='font-size:1.8rem; font-weight:800; color:#2563eb; line-height:1; margin-top:2px;'>#{idx}</div>", unsafe_allow_html=True)
                            with c_body:
                                st.markdown(f"**{clean_question_display(q_text)}**")
                                st.markdown(
                                    f"<span class='unirag-badge unirag-badge-history'>📅 {len(r['years'])} Exam Sittings</span> "
                                    f"<span class='unirag-badge unirag-badge-ranking'>🔄 {r['total_occurrences']} Appearances</span> "
                                    f"<span class='unirag-badge unirag-badge-concept'>🏷️ {r['subject']}</span> "
                                    + (f"<span class='unirag-badge unirag-badge-guidance'>⭐ {m_val} Marks</span>" if m_val else ""),
                                    unsafe_allow_html=True
                                )
                                st.caption(f"**Exam Years:** {y_str} · *Source Papers: {', '.join(r['sample_papers'])}*")

                    # BUNDLED_SOLUTIONS_PATCH: one bundled solutions file for all the questions above
                    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown("<div style='font-weight:700; font-size:1.02rem; color:#0f172a;'>📑 Solutions Bundle: model answers for all the questions above</div>", unsafe_allow_html=True)
                        if answered_count == len(prepared_items):
                            st.caption(f"✅ Solutions for all {len(prepared_items)} questions are included in one file.")
                        elif answered_count > 0:
                            st.caption(f"{answered_count} of {len(prepared_items)} solutions are ready; the rest are marked 'not generated yet' in the file.")
                        else:
                            st.caption("No ready-made solutions for this subject yet. Run: python repeated_solutions.py generate")
                        if answered_count > 0:
                            subj_filename = re.sub(r'[^A-Za-z0-9_]', '_', selected_subject or 'General')
                            try:
                                pdf_bytes = generate_repeated_answers_pdf(prepared_items, subject=selected_subject)
                                st.download_button(
                                    "📥 Download Solutions (PDF)",
                                    data=pdf_bytes,
                                    file_name=f"uniRAG_{subj_filename}_Repeated_Questions_Solutions.pdf",
                                    mime="application/pdf",
                                    key="btn_dl_rep_pdf",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.caption(f"PDF could not be created: {e}")

    with intel_sub3:
        st.caption("Browse past examination questions by their printed marks weightage.")
        _m_all = (not selected_subject) or selected_subject == "All / Auto-detect"
        if _m_all:
            st.markdown(
                """<div class="unirag-empty"><div class="unirag-empty-title">📌 Pick a subject to browse by marks</div>
                <div class="unirag-empty-text">Choose a subject in the <b>Academic Course / Subject Scope</b> dropdown just above the tabs to see its questions grouped by printed marks.</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            _mk_pool = rag._question_bank
            if selected_subject and selected_subject != "All / Auto-detect":
                _mk_pool = [r for r in _mk_pool if r.subject.strip().upper() == selected_subject.strip().upper()] or _mk_pool
            _mk_counts = {}
            for _r in _mk_pool:
                if _r.marks:
                    _mk_counts[_r.marks] = _mk_counts.get(_r.marks, 0) + 1
            _mk_options = [m for m in sorted(_mk_counts, reverse=True) if _mk_counts[m] >= 3] or sorted(_mk_counts, reverse=True) or [10, 6, 5]
            target_m = st.radio("Select Marks Level:", _mk_options, horizontal=True, index=0,
                                format_func=lambda m: f"{m} marks ({_mk_counts.get(m, 0)})", disabled=_m_all)
            if st.button(f"📋 Show {target_m}-Mark Questions", key="btn_show_marks", type="primary", disabled=_m_all):
                with st.spinner(f"Retrieving {target_m}-mark questions..."):
                    q_list = rag.exam_intelligence.get_questions_by_marks(
                        rag._question_bank, marks=target_m, subject=selected_subject, limit=15
                    )
                if not q_list:
                    st.info(f"No {target_m}-mark questions found for {selected_subject}.")
                else:
                    st.write(f"**Found {len(q_list)} question(s) tested for {target_m} marks:**")
                    for i, q in enumerate(q_list, 1):
                        y = q.year or "Year unknown"
                        with st.container(border=True):
                            st.markdown(f"**{i}. [{y}]** {clean_question_display(q.question)}")
                            st.caption(f"📄 Source: *{q.source}* (Page {q.page + 1}) · Marks: **{target_m} Marks**")

# TAB 4: RESOURCE LIBRARY (Unified downloads for Notes, Papers, Syllabus)
with tab_library:
    st.caption("Browse and download official university academic documents: lecture notes, past exam papers, and syllabus.")
    lib_choice = st.radio(
        "Select Resource Category:",
        ["📚 Course Lecture Notes", "📂 Past Exam Question Papers", "📋 Official University Syllabus"],
        horizontal=True,
        label_visibility="collapsed",
        key="resource_lib_selector"
    )

    if lib_choice == "📚 Course Lecture Notes":
        if not NOTES_DIR.exists():
            st.warning("No notes folder found.")
        else:
            pdfs_by_sem_subject = {}
            for pdf in sorted(NOTES_DIR.glob("**/*.pdf")):
                subj = extract_subject(pdf)
                sem = extract_semester(pdf) or "Unknown Semester"
                pdfs_by_sem_subject.setdefault(sem, {}).setdefault(subj, []).append(pdf)

            if not pdfs_by_sem_subject:
                st.info("No lecture notes PDFs found yet.")
            else:
                semesters = sorted(pdfs_by_sem_subject.keys())
                c1, c2 = st.columns(2)
                with c1:
                    chosen_sem = st.selectbox("Semester", semesters, key="lib_notes_sem")
                with c2:
                    subjects_in_sem = sorted(pdfs_by_sem_subject.get(chosen_sem, {}).keys())
                    chosen_subj = st.selectbox("Subject", subjects_in_sem, key="lib_notes_subj")

                if chosen_subj:
                    files = pdfs_by_sem_subject[chosen_sem][chosen_subj]
                    st.write(f"**{len(files)} file(s):**")
                    for pdf in files:
                        with open(pdf, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            f"📄 {pdf.name}", pdf_bytes, file_name=pdf.name,
                            mime="application/pdf", key=f"dl_note_{pdf}"
                        )

    elif lib_choice == "📂 Past Exam Question Papers":
        if not PAPERS_DIR.exists():
            st.warning("No question papers folder found.")
        else:
            papers_by_sem_subject = {}
            for pdf in sorted(PAPERS_DIR.glob("**/*.pdf")):
                subj = extract_subject(pdf)
                sem = extract_semester(pdf) or "Unknown Semester"
                papers_by_sem_subject.setdefault(sem, {}).setdefault(subj, []).append(pdf)

            if not papers_by_sem_subject:
                st.info("No question paper PDFs found yet.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    paper_semesters = sorted(papers_by_sem_subject.keys())
                    chosen_paper_sem = st.selectbox("Semester", paper_semesters, key="lib_papers_sem")
                with c2:
                    subjects_in_paper_sem = sorted(papers_by_sem_subject.get(chosen_paper_sem, {}).keys())
                    chosen_paper_subj = st.selectbox("Subject", subjects_in_paper_sem, key="lib_papers_subj")

                if chosen_paper_subj:
                    paper_files = papers_by_sem_subject[chosen_paper_sem][chosen_paper_subj]
                    st.write(f"**{len(paper_files)} file(s):**")
                    for pdf in paper_files:
                        with open(pdf, "rb") as f:
                            paper_bytes = f.read()
                        st.download_button(
                            f"📄 {pdf.name}", paper_bytes, file_name=pdf.name,
                            mime="application/pdf", key=f"dl_paper_{pdf}"
                        )

    else:
        if not SYLLABUS_DIR.exists():
            st.warning("No syllabus folder found.")
        else:
            syllabus_by_group = {}
            for pdf in sorted(SYLLABUS_DIR.glob("**/*.pdf")):
                group = pdf.parent.name if pdf.parent != SYLLABUS_DIR else "General"
                syllabus_by_group.setdefault(group, []).append(pdf)

            if not syllabus_by_group:
                st.info("No syllabus PDFs found yet.")
            else:
                groups = sorted(syllabus_by_group.keys())
                chosen_group = st.selectbox("Section", groups, key="lib_syllabus_group") if len(groups) > 1 else groups[0]
                syllabus_files = syllabus_by_group[chosen_group]
                st.write(f"**{len(syllabus_files)} file(s):**")
                for pdf in syllabus_files:
                    with open(pdf, "rb") as f:
                        syllabus_bytes = f.read()
                    st.download_button(
                        f"📄 {pdf.name}", syllabus_bytes, file_name=pdf.name,
                        mime="application/pdf", key=f"dl_syl_{pdf}"
                    )

# TAB 5: SETUP (admin only)
if tab_admin is not None:
    with tab_admin:
        st.caption("🛠️ University onboarding — set your university's name and upload your own notes/question papers")

        st.subheader("🏫 University details")
        with st.form("university_settings_form"):
            new_uni_name = st.text_input("University name", value=UNIVERSITY_NAME)
            new_dept = st.text_input("Department", value=DEPARTMENT)
            if st.form_submit_button("💾 Save university details"):
                settings_path = BASE_DIR / "university_settings.json"
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump({"university_name": new_uni_name, "department": new_dept}, f)
                st.success("Saved. Restart the app (or switch role and back) to see the new name everywhere.")

        st.divider()
        st.subheader("👥 Registered students")
        st.caption("Students now register themselves from the login page — this is a read-only view "
                   "of who has signed up so far.")
        existing_students = _load_students()
        if existing_students:
            with st.expander(f"📋 {len(existing_students)} registered student(s)", expanded=True):
                for email, rec in existing_students.items():
                    st.write(f"- {rec.get('name', '(no name)')} — {email}")
        else:
            st.caption("No students have registered yet.")

        st.divider()
        st.subheader("⬆️ Upload a document")
        st.caption("Follows the same (ABBR) Full Name convention used throughout this project — "
                   "give the subject a short abbreviation and its full name.")

        doc_kind = st.selectbox("Document type", ["Notes", "Question Paper"], key="setup_doc_kind")
        col1, col2, col3 = st.columns(3)
        with col1:
            abbr_input = st.text_input("Subject abbreviation (e.g. DBMS)", key="setup_abbr")
        with col2:
            full_name_input = st.text_input("Full subject name (e.g. Database Management Systems)", key="setup_full")
        with col3:
            semester_input = st.number_input("Semester", min_value=1, max_value=8, value=1, step=1, key="setup_sem")

        uploaded_doc = st.file_uploader("Choose a PDF", type=["pdf"], key="setup_upload")

        if uploaded_doc and abbr_input and full_name_input:
            if st.button("💾 Save document to disk"):
                def _clean_name(s):
                    # strip path separators / Windows-illegal characters and ".." (HARDENING_PATCH_APPLIED)
                    s = re.sub(r'[\\/:*?"<>|]+', "", s).replace("..", "")
                    return s.strip().strip(".")
                safe_abbr = _clean_name(abbr_input).upper()
                safe_full = _clean_name(full_name_input)
                if not safe_abbr or not safe_full:
                    st.error("Abbreviation / full name contain only characters that are not allowed in folder names.")
                    st.stop()
                if doc_kind == "Notes":
                    folder = NOTES_DIR / f"SEM{semester_input}" / f"({safe_abbr}) {safe_full}"
                    folder.mkdir(parents=True, exist_ok=True)
                    dest = folder / Path(str(uploaded_doc.name).replace("\\", "/")).name
                else:
                    folder = PAPERS_DIR / f"SEM{semester_input}"
                    folder.mkdir(parents=True, exist_ok=True)
                    original_stem = Path(uploaded_doc.name).stem
                    dest = folder / f"({safe_abbr}) {safe_full} - {original_stem}.pdf"

                with open(dest, "wb") as f:
                    f.write(uploaded_doc.getbuffer())
                st.success(f"Saved to {dest}")
                st.info("Uploaded documents only take effect after a rebuild — use the button below once "
                        "you've uploaded everything you need.")

        st.divider()
        st.subheader("🔄 Rebuild index")
        st.caption("Re-processes every notes/question-paper PDF and rebuilds all FAISS indexes. "
                   "This can take a long time on a large corpus — the app will be unresponsive "
                   "while it runs. Only do this once after uploading new documents, not per-file.")
        if st.button("🔄 Rebuild everything now", type="primary"):
            with st.spinner("Rebuilding — this may take a while..."):
                result = rag.rebuild_everything()
            st.success(f"Rebuilt {result['subjects']} subject indexes, {result['paper_chunks']} question-paper chunks.")
            st.info("Subject dropdown and all data are now live — no restart needed.")

# FOOTER
st.divider()
st.caption("🔒 100% Private • Powered by Ollama + FAISS")