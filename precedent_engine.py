# ===========================================================
# ✅ ADVANCED LEGAL PRECEDENT ENGINE (MODE C - SAFE)
# ===========================================================

import re
import json
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sentence_transformers import SentenceTransformer, util

Base = declarative_base()


# ===========================================================
# ✅ 1. DATABASE MODEL
# ===========================================================
class LegalPrecedent(Base):
    __tablename__ = "legal_precedents"

    id = Column(String, primary_key=True)
    case_name = Column(String, nullable=False)
    citation = Column(String, nullable=False)
    court = Column(String, nullable=False)
    year = Column(Integer)
    summary = Column(Text)
    legal_principles = Column(JSON)
    keywords = Column(JSON)
    full_text = Column(Text)
    importance_score = Column(Integer)
    jurisdiction = Column(String)
    embedding = Column(JSON)  # ✅ vector embedding


# ===========================================================
# ✅ 2. MAIN ENGINE
# ===========================================================
class PrecedentEngine:
    def __init__(self, db_path="precedents.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # ✅ Pre-populate landmark cases if DB is empty
        if self.session.query(LegalPrecedent).count() == 0:
            self._load_landmark_cases()

    # -------------------------------------------------------
    # ✅ Load Important Indian Landmark Cases
    # -------------------------------------------------------
    def _load_landmark_cases(self):
        landmark = [
            {
                "id": "kesavananda_1973",
                "case_name": "Kesavananda Bharati v. State of Kerala",
                "citation": "(1973) 4 SCC 225",
                "court": "Supreme Court",
                "year": 1973,
                "summary": "Established the Basic Structure doctrine...",
                "legal_principles": ["Basic structure", "Judicial review"],
                "keywords": ["constitutional amendment", "basic structure"],
                "importance_score": 10,
                "jurisdiction": "India"
            },
            {
                "id": "maneka_1978",
                "case_name": "Maneka Gandhi v. Union of India",
                "citation": "AIR 1978 SC 597",
                "court": "Supreme Court",
                "year": 1978,
                "summary": "Expanded Article 21 to include fairness, justice...",
                "legal_principles": ["Due process", "Right to life"],
                "keywords": ["article 21", "liberty", "due process"],
                "importance_score": 9,
                "jurisdiction": "India"
            }
        ]

        for c in landmark:
            emb = self.embedder.encode(c["summary"]).tolist()
            prec = LegalPrecedent(embedding=emb, **c)
            self.session.add(prec)

        self.session.commit()
        print("✅ Loaded landmark Supreme Court precedents")

    # -------------------------------------------------------
    # ✅ Add New Precedent
    # -------------------------------------------------------
    def add_precedent(self, case_data: Dict):
        case_data["embedding"] = (
            self.embedder.encode(case_data["summary"]).tolist()
            if "summary" in case_data else None
        )
        self.session.add(LegalPrecedent(**case_data))
        self.session.commit()

    # -------------------------------------------------------
    # ✅ Indian Kanoon Scraper (Safe Metadata ONLY)
    # -------------------------------------------------------
    def fetch_from_indian_kanoon(self, query: str, max_results=5):
        """
        ✅ Safe: Extracts searchable metadata only (no legal advice)
        ✅ Uses public web results
        """
        url = f"https://indiankanoon.org/search/?formInput={query}"

        try:
            r = requests.get(url, timeout=6)
            html = r.text

            pattern = r'<a href="(/doc/\d+)">([^<]+)</a>'
            matches = re.findall(pattern, html)

            results = []
            for link, title in matches[:max_results]:
                results.append({
                    "title": title,
                    "url": "https://indiankanoon.org" + link
                })

            return results

        except Exception:
            return []

    # -------------------------------------------------------
    # ✅ Semantic Search
    # -------------------------------------------------------
    def semantic_search(self, query: str, top_k=5):
        q_emb = self.embedder.encode(query)

        cases = self.session.query(LegalPrecedent).all()
        scored = []

        for c in cases:
            if not c.embedding:
                continue

            # ✅ convert stored embedding to float32
            emb = np.array(c.embedding, dtype=np.float32)

            # ✅ compute cosine similarity safely
            sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))


            scored.append((c, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


    # -------------------------------------------------------
    # ✅ Hybrid Search: keyword + semantic
    # -------------------------------------------------------
    def search(self, query: str):
        # 1. Semantic
        sem = self.semantic_search(query, top_k=7)

        # 2. Keyword matches
        key = self.session.query(LegalPrecedent).filter(
            LegalPrecedent.summary.ilike(f"%{query}%")
        ).all()

        return {
            "semantic_matches": [(c.case_name, score) for c, score in sem],
            "text_matches": [c.case_name for c in key]
        }

    # -------------------------------------------------------
    # ✅ Relevance Scoring
    # -------------------------------------------------------
    def score_precedent(self, precedent, query: str, sim_score: float):
        score = 0

        # ✅ 50% semantic similarity
        score += sim_score * 5

        # ✅ 30% importance
        score += (precedent.importance_score or 5) * 0.3

        # ✅ 20% recency
        if precedent.year:
            age = datetime.now().year - precedent.year
            score += max(0, (20 - age) / 20) * 2

        return round(score, 2)

    # -------------------------------------------------------
    # ✅ Human-Like Reasoning Engine (Mode C)
    # -------------------------------------------------------
    def explain_relevance(self, precedent, query: str) -> List[str]:
        explanations = []

        q = query.lower()
        summ = (precedent.summary or "").lower()

        if any(k in q for k in (precedent.keywords or [])):
            explanations.append(
                "The factual background appears to involve similar legal keywords."
            )

        if any(p.lower() in q for p in (precedent.legal_principles or [])):
            explanations.append(
                "The legal issue engages principles discussed in this judgment."
            )

        if precedent.court.lower() == "supreme court":
            explanations.append("Supreme Court precedents carry high persuasive value.")

        if not explanations:
            explanations.append(
                "There is conceptual similarity based on the legal issue and the case summary."
            )

        return explanations

    # -------------------------------------------------------
    # ✅ MAIN INTERFACE: Get Ranked Relevant Precedents
    # -------------------------------------------------------
    def find_precedents(self, query: str, top_k=5):
        semantic = self.semantic_search(query, top_k=top_k)

        results = []
        for case, sim in semantic:
            score = self.score_precedent(case, query, sim)
            reasoning = self.explain_relevance(case, query)

            results.append({
                "case_name": case.case_name,
                "citation": case.citation,
                "court": case.court,
                "year": case.year,
                "summary": case.summary,
                "similarity_score": round(sim, 3),
                "final_score": score,
                "reasoning": reasoning
            })

        return results

    # -------------------------------------------------------
    # ✅ Context-Aware Report Generator (No Legal Advice)
    # -------------------------------------------------------
    def generate_reasoning_report(self, query: str):
        precedents = self.find_precedents(query, top_k=3)

        report = {
            "issue_identified": query,
            "analysis": [],
            "suggested_considerations": [
                "Whether the factual matrix aligns with the legal principles identified.",
                "The jurisdiction and hierarchy of cited precedents.",
                "Whether similar procedural or constitutional issues arise."
            ]
        }

        for p in precedents:
            report["analysis"].append({
                "case": p["case_name"],
                "citation": p["citation"],
                "summary": p["summary"],
                "why_relevant": p["reasoning"],
                "score": p["final_score"]
            })

        return report
