import json
import re
import sqlite3
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
PROJECT = ROOT 
DB_PATH = PROJECT / "precedents.db"

st.set_page_config(page_title="LegalBERT India | Research Demo", page_icon="⚖️", layout="wide")

st.title("⚖️ LegalBERT India")
st.caption("Indian legal-document research prototype — not legal advice")

st.warning(
    "Research prototype only. Results may be incomplete or inaccurate and must be verified "
    "against authoritative legal sources by a qualified legal professional. Do not upload "
    "confidential, privileged, or personally sensitive documents."
)

def extract_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix == ".txt":
        return raw.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(raw))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise RuntimeError(f"Could not read this PDF: {exc}") from exc
    if suffix == ".docx":
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            raise RuntimeError(f"Could not read this DOCX: {exc}") from exc
    raise RuntimeError("Supported formats are PDF, DOCX, and TXT.")

def extract_entities(text):
    patterns = {
        "Sections": r"\b(?:Section|Sec\.?)\s+\d+[A-Za-z()/-]*",
        "Acts": r"\b[A-Z][A-Za-z ,&()'-]{2,80}\bAct,?\s+\d{4}\b",
        "Dates": r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        "Case references": r"\b[A-Z][A-Za-z.& ]{1,70}\s+v\.?\s+[A-Z][A-Za-z.& ]{1,70}",
        "Money": r"(?:₹|Rs\.?\s*)\s?\d[\d,]*(?:\.\d{1,2})?"
    }
    return {label: sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))[:100]
            for label, pattern in patterns.items()}

def classify_clauses(text):
    groups = {
        "Payment": ["payment", "fee", "invoice", "consideration"],
        "Termination": ["terminate", "termination", "expire", "cancel"],
        "Liability / indemnity": ["liable", "liability", "damages", "indemnify", "indemnity"],
        "Confidentiality": ["confidential", "non-disclosure", "non disclosure"],
        "Dispute resolution": ["arbitration", "arbitral tribunal", "jurisdiction", "dispute"],
        "Data protection": ["personal data", "data breach", "privacy", "security practices"],
    }
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text) if len(p.strip()) > 30]
    out = []
    for paragraph in paragraphs[:100]:
        low = paragraph.lower()
        matched = [name for name, words in groups.items() if any(word in low for word in words)]
        if matched:
            out.append({"category": ", ".join(matched), "text": paragraph[:1200]})
    return out

def search_precedents(query):
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2][:8]
        if not words:
            return []
        conditions = " OR ".join(["LOWER(COALESCE(case_name,'') || ' ' || COALESCE(summary,'') || ' ' || COALESCE(citation,'')) LIKE ?" for _ in words])
        params = [f"%{w.lower()}%" for w in words]
        rows = con.execute(
            f"SELECT case_name, citation, court, year, summary, jurisdiction FROM legal_precedents WHERE {conditions} LIMIT 10",
            params
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()

tab_doc, tab_search, tab_about = st.tabs(["Document analysis", "Precedent lookup", "About & limitations"])

with tab_doc:
    uploaded = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
    if uploaded:
        try:
            text = extract_text(uploaded)
            if not text.strip():
                st.error("No selectable text was found. Scanned PDFs need OCR, which is not enabled in this first deployment version.")
            else:
                st.subheader("Extracted text")
                st.text_area("Document text", text, height=220)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Pattern-based entity hints")
                    entities = extract_entities(text)
                    for label, values in entities.items():
                        with st.expander(f"{label} ({len(values)})", expanded=bool(values)):
                            st.write(values or "No matches found.")
                with c2:
                    st.subheader("Keyword-based clause hints")
                    clauses = classify_clauses(text)
                    if clauses:
                        for item in clauses:
                            with st.expander(item["category"]):
                                st.write(item["text"])
                    else:
                        st.info("No clauses matched the prototype's keyword rules.")
                st.caption("These are regex/keyword hints, not output from your fine-tuned LegalBERT model.")
        except Exception as exc:
            st.error(str(exc))

with tab_search:
    st.write("Search the SQLite precedent records included in this project. The current database contains only a small demo set.")
    query = st.text_input("Legal issue or keywords", placeholder="e.g. Article 21 right to life")
    if st.button("Search precedents", type="primary"):
        results = search_precedents(query)
        if not results:
            st.info("No matching records found in the current local database.")
        for row in results:
            st.markdown(f"**{row.get('case_name') or 'Untitled case'}**")
            st.caption(f"{row.get('citation') or 'Citation unavailable'} · {row.get('court') or 'Court unavailable'} · {row.get('year') or 'Year unavailable'}")
            st.write(row.get("summary") or "No summary stored.")
            st.divider()

with tab_about:
    st.markdown("""
    **Current prototype scope**
    - Upload and extract text from text-based PDF, DOCX, and TXT files.
    - Surface pattern-based references such as sections, acts, dates, money, and case names.
    - Categorize paragraphs using keyword rules.
    - Search the SQLite records shipped with the repository.

    **Not yet integrated**
    - Fine-tuned LegalBERT inference: the uploaded archive contains a model config but no model weights/tokenizer files.
    - OCR for scanned documents.
    - Full jurisdiction mapping and a production risk-assessment workflow.
    - A comprehensive, verified precedent corpus.

    Do not treat prototype classifications or matches as legal advice. Verify case citations, current law, jurisdiction, and all outputs independently.
    """)
