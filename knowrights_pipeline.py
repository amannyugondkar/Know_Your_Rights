# =============================================================================
#  KnowRights AI - Unified Pipeline
#  India-focused Legal Assistant
# =============================================================================

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq").lower()
VECTIFY_API_URL = os.getenv("VECTIFY_API_URL", "").rstrip("/")
VECTIFY_API_KEY = os.getenv("VECTIFY_API_KEY", "")

SUPPORTED_LANGS: Dict[str, str] = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "te": "Telugu",  "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

RiskLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
RiskClass = Literal["HIGH_RISK", "MODERATE_RISK", "LOW_RISK", "MISSING_CLAUSE", "OK"]

@dataclass
class DetectedRisk:
    level:        RiskLevel
    category:     str
    description:  str
    suggestion:   str
    matched_text: str = ""

@dataclass
class ClauseRisk:
    clause:      str
    risk_score:  int
    risk_class:  RiskClass
    explanation: str
    negotiation: str = ""

@dataclass
class FraudSignal:
    signal:     str
    confidence: float
    detail:     str

@dataclass
class ContractAnalysisResult:
    raw_text:        str
    language:        str
    translated_text: str
    safety_score:    int
    safety_level:    str
    risks:           List[DetectedRisk] = field(default_factory=list)
    clauses:         List[ClauseRisk]   = field(default_factory=list)
    fraud_signals:   List[FraudSignal]  = field(default_factory=list)
    missing_clauses: List[str]          = field(default_factory=list)
    ai_summary:      str                = ""
    simple_summary:  str                = ""
    report:          dict               = field(default_factory=dict)

@dataclass
class CaseAssessment:
    query:               str
    language:            str
    translated_query:    str
    relevant_laws:       List[str]
    win_probability:     float
    risk_level:          str
    reasoning:           str
    recommended_actions: List[str] = field(default_factory=list)
    simple_explanation:  str       = ""

@dataclass
class ChatMessage:
    role:    Literal["user", "assistant"]
    content: str

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LANGUAGE UTILS
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"

def translate_text(text: str, src: str, tgt: str) -> str:
    if src == tgt or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator
        if len(text) <= 4500:
            return GoogleTranslator(source=src, target=tgt).translate(text)
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        return " ".join(GoogleTranslator(source=src, target=tgt).translate(c) for c in chunks)
    except Exception:
        return text

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DOCUMENT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_bytes(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber, io
            parts = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n\n".join(parts)
        except Exception as e:
            return f"[PDF extraction failed: {e}]"
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LEGAL KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

STATIC_KB: List[Dict] = [
    {"title": "ICA §10 – Valid Contracts",
     "text":  "Agreements are contracts if made by free consent of competent parties for lawful consideration with a lawful object."},
    {"title": "ICA §14 – Free Consent",
     "text":  "Consent is free when not caused by coercion, undue influence, fraud, misrepresentation, or mistake."},
    {"title": "ICA §15 – Coercion",
     "text":  "Coercion is committing or threatening to commit an act forbidden by the IPC, or unlawful detention of property."},
    {"title": "ICA §17 – Fraud",
     "text":  "Fraud includes false representation, active concealment, promise with no intent to perform, and other deceptive acts."},
    {"title": "ICA §19 – Voidable Contracts",
     "text":  "Contracts caused by coercion, fraud, or misrepresentation are voidable at the option of the aggrieved party."},
    {"title": "ICA §23 – Unlawful Consideration",
     "text":  "Consideration or object is unlawful if forbidden by law, fraudulent, harmful to persons/property, or against public policy."},
    {"title": "ICA §73 – Breach Compensation",
     "text":  "Party suffering from breach is entitled to compensation for loss or damage naturally arising from the breach."},
    {"title": "ICA §74 – Penalty Clauses",
     "text":  "Courts award only reasonable compensation not exceeding the stipulated penalty, regardless of proof of actual loss."},
    {"title": "ICA §27 – Non-Compete",
     "text":  "Every agreement that restrains a person from carrying on a lawful profession or trade is void to that extent."},
    {"title": "Motor Vehicles Act §166",
     "text":  "Claim for compensation may be made by the person who sustained injury or the owner of damaged property or in fatal cases by legal representatives."},
    {"title": "Motor Vehicles Act §140",
     "text":  "Liability to pay compensation in certain cases on the principle of no fault; compensation of Rs 50,000 for death."},
    {"title": "Consumer Protection Act §2",
     "text":  "Consumers can file complaints about defective goods, deficient services, unfair trade practices, and restrictive trade practices."},
    {"title": "IT Act §66C – Identity Theft",
     "text":  "Whoever fraudulently uses the electronic signature, password, or any other unique identification feature of another person shall be punished."},
    {"title": "IT Act §66D – Phishing",
     "text":  "Cheating by personation using computer resources is punishable with imprisonment up to 3 years and fine up to 1 lakh."},
    {"title": "Mohori Bibee v. Dharmodas Ghose (1903)",
     "text":  "A contract with a minor is void ab initio and cannot be ratified even on attaining majority."},
    {"title": "Hadley v. Baxendale (1854)",
     "text":  "Damages limited to those that arise naturally from the breach or were reasonably foreseeable at contract formation."},
    {"title": "Fateh Chand v. Balkishan Dass (1963)",
     "text":  "Penalty clauses are unenforceable beyond reasonable compensation under Indian law."},
    {"title": "NDPS Act §20",
     "text":  "Possession, sale, purchase, transport of narcotic drugs is a criminal offence with stringent penalties."},
    {"title": "IPC §420 – Cheating",
     "text":  "Whoever cheats and thereby dishonestly induces the person deceived to deliver property shall be punished with imprisonment up to 7 years."},
]

_rag_index    = None
_rag_embedder = None

def _build_rag_index():
    global _rag_index, _rag_embedder
    if _rag_index is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        import faiss, numpy as np
        _rag_embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embs = _rag_embedder.encode([d["text"] for d in STATIC_KB], show_progress_bar=False)
        _rag_index = faiss.IndexFlatL2(embs.shape[1])
        _rag_index.add(embs.astype("float32"))
    except Exception:
        _rag_index = None

def retrieve_context(query: str, top_k: int = 4) -> List[str]:
    if VECTIFY_API_URL and VECTIFY_API_KEY:
        try:
            import requests as req
            resp = req.post(
                f"{VECTIFY_API_URL}/retrieve",
                json={"query": query, "top_k": top_k},
                headers={"Authorization": f"Bearer {VECTIFY_API_KEY}"},
                timeout=20,
            )
            resp.raise_for_status()
            snippets = [
                item.get("text") or item.get("content") or ""
                for item in resp.json().get("results", [])
            ]
            if snippets:
                return [s for s in snippets if s][:top_k]
        except Exception:
            pass
    _build_rag_index()
    if _rag_index is not None and _rag_embedder is not None:
        try:
            import numpy as np
            q_emb = _rag_embedder.encode([query]).astype("float32")
            _, idx = _rag_index.search(q_emb, top_k)
            return [f"{STATIC_KB[i]['title']}: {STATIC_KB[i]['text']}" for i in idx[0]]
        except Exception:
            pass
    return [f"{d['title']}: {d['text']}" for d in STATIC_KB[:top_k]]

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RISK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

RISK_PATTERNS: List[Dict] = [
    {"level": "CRITICAL", "category": "Unlimited Liability",
     "patterns": [r"unlimited\s+liability", r"indemnify.*?all\s+claims", r"liable\s+for\s+all\s+losses"],
     "description": "Exposes you to unlimited financial risk with no cap.",
     "suggestion":  "Negotiate a cap at the contract value or a fixed amount."},
    {"level": "CRITICAL", "category": "Irrevocable / Cannot Exit",
     "patterns": [r"irrevocable", r"cannot\s+be\s+terminated", r"perpetual\s+obligation"],
     "description": "Locks you in permanently with no exit.",
     "suggestion":  "Add a termination-for-convenience clause with 30-day notice."},
    {"level": "CRITICAL", "category": "Waiver of Legal Rights",
     "patterns": [r"waive.*?right\s+to\s+sue", r"no\s+right\s+to\s+damages", r"waive.*?claims"],
     "description": "You give up the right to take legal action.",
     "suggestion":  "Never waive legal rights entirely; seek arbitration instead."},
    {"level": "HIGH", "category": "Unilateral Termination",
     "patterns": [r"terminate.*?without\s+cause", r"sole\s+discretion", r"at\s+any\s+time\s+without\s+notice"],
     "description": "Other party can end the contract anytime without reason.",
     "suggestion":  "Request 30-day notice and severance terms."},
    {"level": "HIGH", "category": "Broad Non-Compete",
     "patterns": [r"non.?compete", r"shall\s+not\s+compete", r"restrictive\s+covenant"],
     "description": "Restricts your ability to work elsewhere.",
     "suggestion":  "Limit scope to 1-2 years and specific geography under ICA §27."},
    {"level": "HIGH", "category": "Unilateral Modification",
     "patterns": [r"reserves\s+the\s+right\s+to\s+modify", r"may\s+amend.*?without\s+notice"],
     "description": "Other party can change contract terms without your consent.",
     "suggestion":  "Require mutual written consent for any amendments."},
    {"level": "HIGH", "category": "Excessive Penalty",
     "patterns": [r"penalty.*?exceed", r"liquidated\s+damages.*?(?:5|10|20)\s*(?:x|times|%)", r"forfeit.*?entire"],
     "description": "Penalty clause may be unenforceable but still coercive.",
     "suggestion":  "Cite ICA §74; cap penalties at actual proven loss."},
    {"level": "MEDIUM", "category": "Auto-Renewal",
     "patterns": [r"automatically\s+renew", r"auto.?renew", r"unless\s+cancelled\s+in\s+writing"],
     "description": "Contract renews itself unless you act.",
     "suggestion":  "Set a calendar reminder 60 days before renewal."},
    {"level": "MEDIUM", "category": "IP Assignment",
     "patterns": [r"intellectual\s+property.*?(?:belong|vest|assign)", r"work\s+for\s+hire"],
     "description": "Your creations automatically belong to the other party.",
     "suggestion":  "Clarify scope; retain rights to prior IP and personal projects."},
    {"level": "MEDIUM", "category": "Governing Law / Foreign Jurisdiction",
     "patterns": [r"governed\s+by.*?(?:delaware|uk|singapore|new\s+york)\s+law",
                  r"jurisdiction.*?(?:foreign|outside\s+india)"],
     "description": "Disputes resolved under foreign law, costly for you.",
     "suggestion":  "Negotiate for Indian courts under Indian law."},
    {"level": "LOW", "category": "Confidentiality",
     "patterns": [r"confidential", r"non.?disclosure"],
     "description": "Restricts sharing of information.",
     "suggestion":  "Verify duration and scope of confidentiality obligations."},
    {"level": "LOW", "category": "Force Majeure",
     "patterns": [r"force\s+majeure", r"act\s+of\s+god"],
     "description": "Standard clause excusing performance in extreme events.",
     "suggestion":  "Check it includes epidemics, government orders, etc."},
]

MISSING_CLAUSE_CHECKS: List[Tuple[str, str]] = [
    (r"dispute\s+resolution|arbitration|mediation",
     "Dispute Resolution - No mechanism for resolving disagreements."),
    (r"termination",
     "Termination Clause - No clear exit procedure defined."),
    (r"payment\s+terms|invoice|due\s+date",
     "Payment Terms - Payment schedule / due dates not specified."),
    (r"limitation\s+of\s+liability|liability.*?cap",
     "Liability Cap - No ceiling on financial exposure."),
    (r"governing\s+law|jurisdiction",
     "Governing Law - Applicable law / jurisdiction not stated."),
    (r"amendment|modification\s+clause",
     "Amendment Clause - No process for changing the contract."),
    (r"warranty|representation",
     "Warranties & Representations - No guarantees about the subject matter."),
]

FRAUD_SIGNALS: List[Dict] = [
    {"signal": "Vague / No Consideration",
     "pattern": r"free\s+of\s+charge|no\s+consideration|gratuitous",
     "confidence": 0.6,
     "detail": "Lack of clear consideration raises enforceability concerns."},
    {"signal": "Pressure Clause",
     "pattern": r"sign\s+immediately|expires\s+today|limited\s+time\s+only",
     "confidence": 0.75,
     "detail": "Pressure to sign quickly is a red flag for fraudulent contracts."},
    {"signal": "Phantom Identity",
     "pattern": r"company\s+reg(?:istration)?\s+(?:no|number)?\s*:?\s*[xX?]{4,}|tbd|to\s+be\s+decided",
     "confidence": 0.8,
     "detail": "Incomplete party identification suggests a fraudulent entity."},
    {"signal": "Unconscionable Terms",
     "pattern": r"no\s+recourse|non-refundable.*?under\s+any\s+circumstances|absolute.*?forfeit",
     "confidence": 0.7,
     "detail": "Extremely one-sided terms may render the contract unconscionable."},
    {"signal": "Illegal Object",
     "pattern": r"narcotic|drug\s+traf|launder|bribe|kickback",
     "confidence": 0.95,
     "detail": "Contract references potentially illegal activities - void under ICA §23."},
]

_RISK_WEIGHTS: Dict[str, int] = {"CRITICAL": 35, "HIGH": 20, "MEDIUM": 10, "LOW": 3}
_FRAUD_BASE_PENALTY = 25

def run_risk_engine(text: str) -> Tuple[List[DetectedRisk], List[str], List[FraudSignal], int]:
    text_lower = text.lower()
    risks: List[DetectedRisk] = []
    seen_cats: set = set()

    for pg in RISK_PATTERNS:
        if pg["category"] in seen_cats:
            continue
        for pat in pg["patterns"]:
            m = re.search(pat, text_lower)
            if m:
                risks.append(DetectedRisk(
                    level        = pg["level"],
                    category     = pg["category"],
                    description  = pg["description"],
                    suggestion   = pg["suggestion"],
                    matched_text = text[max(0, m.start()-40): m.end()+40].strip(),
                ))
                seen_cats.add(pg["category"])
                break

    missing: List[str] = []
    for pat, label in MISSING_CLAUSE_CHECKS:
        if not re.search(pat, text_lower):
            missing.append(label)

    fraud_signals: List[FraudSignal] = []
    for fs in FRAUD_SIGNALS:
        if re.search(fs["pattern"], text_lower):
            fraud_signals.append(FraudSignal(
                signal     = fs["signal"],
                confidence = fs["confidence"],
                detail     = fs["detail"],
            ))

    raw_score  = sum(_RISK_WEIGHTS.get(r.level, 0) for r in risks)
    raw_score += len(missing) * 5
    raw_score += sum(int(f.confidence * _FRAUD_BASE_PENALTY) for f in fraud_signals)
    raw_score  = min(100, raw_score)

    return risks, missing, fraud_signals, raw_score


def rl_policy_score(raw_score: int, fraud_signals: List[FraudSignal]) -> Tuple[int, str]:
    fraud_penalty = sum(f.confidence * 15 for f in fraud_signals)
    adjusted = min(100, raw_score + fraud_penalty)
    safety   = max(0, 100 - int(adjusted))

    if safety >= 80:
        label = "LOW RISK"
    elif safety >= 55:
        label = "MODERATE RISK"
    elif safety >= 30:
        label = "HIGH RISK"
    else:
        label = "EXTREME RISK - DO NOT SIGN"

    return safety, label

# ─────────────────────────────────────────────────────────────────────────────
# 7.  LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self) -> None:
        self.provider = LLM_PROVIDER
        self._openai  = None

        if self.provider == "groq" and GROQ_API_KEY:
            pass  # validated at call time
        elif self.provider == "gemini" and GEMINI_API_KEY:
            try:
                from google import genai
                genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                print(f"  [Gemini init error: {e}]")
                self.provider = None
        elif self.provider == "openai" and OPENAI_API_KEY:
            try:
                import openai
                openai.api_key = OPENAI_API_KEY
                self._openai = openai
            except Exception as e:
                print(f"  [OpenAI init error: {e}]")
                self.provider = None
        else:
            self.provider = None

    def _call(self, prompt: str, max_tokens: int = 800) -> str:
        if self.provider == "groq" and GROQ_API_KEY:
            try:
                import requests
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a legal assistant specializing in Indian law."},
                            {"role": "user",   "content": prompt},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.3,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"  [Groq error: {e}]")
                return ""

        if self.provider == "gemini" and GEMINI_API_KEY:
            try:
                from google import genai
                client   = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                return (response.text or "").strip()
            except Exception as e:
                print(f"  [Gemini error: {e}]")
                return ""

        if self.provider == "openai" and self._openai:
            try:
                res = self._openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a legal assistant for India."},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return res.choices[0].message["content"].strip()
            except Exception as e:
                print(f"  [OpenAI error: {e}]")
                return ""

        return ""

    def analyze_contract(self, text: str, risks: List[DetectedRisk],
                         missing: List[str], fraud: List[FraudSignal]) -> str:
        risk_summary  = "\n".join(f"- [{r.level}] {r.category}: {r.description}" for r in risks[:6]) or "None detected."
        fraud_summary = "\n".join(f"- {f.signal} (conf {f.confidence:.0%}): {f.detail}" for f in fraud) or "None."
        missing_str   = "\n".join(f"- {m}" for m in missing) or "None."

        prompt = f"""You are a senior Indian contract lawyer.

CONTRACT TEXT (excerpt):
{text[:2500]}

RISK ENGINE FINDINGS:
Risks:
{risk_summary}

Missing Clauses:
{missing_str}

Fraud Signals:
{fraud_summary}

Write a structured contract analysis with:
1. Overall assessment (2-3 sentences)
2. Top 3 most dangerous clauses and why
3. Missing protections the signer should insist on
4. Whether this contract appears legitimate or suspicious
5. Recommended next steps

Be specific, cite Indian law where relevant. Keep under 400 words.
End with: Always consult a qualified lawyer before signing."""
        return self._call(prompt) or self._fallback_analysis(risks, missing, fraud)

    def simplify_analysis(self, analysis: str) -> str:
        if not analysis:
            return ""
        prompt = f"""Explain this legal analysis like I am 10 years old.
Use simple words, short sentences, and an analogy a child would understand.
3-5 sentences max.

Analysis:
{analysis[:1500]}"""
        return self._call(prompt, max_tokens=200) or (
            "This contract has some parts that might not be fair to you. "
            "Some rules could make you pay a lot of money or stop you from doing things you want. "
            "It is like a game where the other person wrote all the rules. Ask a grown-up expert before signing!"
        )

    def assess_case_query(self, query: str, context: List[str]) -> Dict:
        ctx    = "\n\n".join(context[:4])
        prompt = f"""You are an AI legal information assistant for Indian citizens.

USER QUERY: {query}

RELEVANT LAWS / PRECEDENTS:
{ctx}

You MUST respond with ONLY a raw JSON object. No markdown, no backticks, no explanation before or after.
Start your response with {{ and end with }}

Example format:
{{"win_probability": 0.7, "risk_level": "MEDIUM", "relevant_laws": ["Motor Vehicles Act §166", "Motor Vehicles Act §140"], "reasoning": "Your reasoning here in 3-5 sentences.", "recommended_actions": ["Action 1", "Action 2", "Action 3"], "simple_explanation": "Simple explanation here."}}

Now respond for this query with the same JSON format:"""

        raw = self._call(prompt, max_tokens=600)

        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except Exception:
            pass
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {
            "win_probability": 0.5,
            "risk_level": "MEDIUM",
            "relevant_laws": ["Motor Vehicles Act §166", "Indian Contract Act"],
            "reasoning": raw or "Unable to generate structured assessment. Please consult a lawyer.",
            "recommended_actions": [
                "Consult a qualified lawyer.",
                "Document all evidence.",
                "File a complaint with the appropriate authority.",
            ],
            "simple_explanation": "This is a complex situation. Speaking with a lawyer is the best first step.",
        }

    def explain_clause(self, clause: str) -> Tuple[str, str]:
        prompt = f"""For this legal clause from an Indian contract:

CLAUSE: {clause[:800]}

Respond with a JSON object:
{{
  "simple": "<explain in 2 sentences for a non-lawyer>",
  "negotiation": "<one practical negotiation tip citing Indian law if possible>"
}}"""
        raw = self._call(prompt, max_tokens=250)
        try:
            data = json.loads(re.sub(r"```(?:json)?|```", "", raw).strip())
            return data.get("simple", ""), data.get("negotiation", "")
        except Exception:
            return ("This clause defines responsibilities and consequences.", "Ask for a more balanced version.")

    def chat_answer(self, question: str, history: List[ChatMessage],
                    context: List[str], contract_summary: str = "") -> str:
        ctx_str      = "\n\n".join(context[:4])
        hist_str     = "\n".join(f"{m.role.upper()}: {m.content}" for m in history[-4:])
        contract_ctx = f"\nCONTRACT CONTEXT:\n{contract_summary[:800]}\n" if contract_summary else ""

        prompt = f"""You are KnowRights, an AI legal information assistant for Indian citizens.
You are NOT a lawyer and cannot give official legal advice.
{contract_ctx}
RELEVANT LAWS:
{ctx_str}

CONVERSATION HISTORY:
{hist_str}

USER: {question}

Answer in clear, simple language (3-5 short paragraphs).
Cite specific Indian law sections where helpful.
End with: Please consult a qualified lawyer or free legal aid clinic for your specific situation."""
        return self._call(prompt, max_tokens=600) or (
            "I am unable to retrieve an AI-generated answer right now. "
            "Please contact a qualified lawyer or your nearest legal aid clinic."
        )

    def translate_explanation(self, text: str, language: str) -> str:
        if not text:
            return ""
        prompt = f"Translate this legal explanation into {language}. Keep it simple and respectful.\n\n{text[:1500]}"
        return self._call(prompt, max_tokens=600) or translate_text(text, "en", language[:2].lower())

    @staticmethod
    def _fallback_analysis(risks, missing, fraud) -> str:
        lines = ["Contract Analysis (offline mode):\n"]
        if risks:
            lines.append("RISKS DETECTED:")
            for r in risks[:5]:
                lines.append(f"  [{r.level}] {r.category} - {r.description}")
                lines.append(f"    -> {r.suggestion}")
        if missing:
            lines.append("\nMISSING CLAUSES:")
            for m in missing:
                lines.append(f"  - {m}")
        if fraud:
            lines.append("\nFRAUD SIGNALS:")
            for f in fraud:
                lines.append(f"  ! {f.signal}: {f.detail}")
        lines.append("\nAlways consult a qualified lawyer before signing.")
        return "\n".join(lines)


llm = LLMClient()

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_contract_pipeline(
    text: str,
    filename: str = "contract.txt",
    response_language: str = "en",
) -> ContractAnalysisResult:
    print("  [1/6] Detecting language...")
    lang = detect_language(text)

    print(f"  [2/6] Translating from '{lang}' to English...")
    en_text = translate_text(text, lang, "en") if lang != "en" else text

    print("  [3/6] Running risk engine + RL scoring...")
    risks, missing, fraud, raw_score = run_risk_engine(en_text)
    safety_score, safety_level = rl_policy_score(raw_score, fraud)

    print("  [4/6] Retrieving legal context (VectifyAI / FAISS)...")
    query_for_rag    = " ".join(r.category for r in risks[:3]) or "Indian contract law"
    context_snippets = retrieve_context(query_for_rag, top_k=4)

    print("  [5/6] Running AI analysis...")
    ai_summary     = llm.analyze_contract(en_text, risks, missing, fraud)
    simple_summary = llm.simplify_analysis(ai_summary)

    if response_language != "en":
        print(f"  [5b] Translating output to '{response_language}'...")
        ai_summary     = llm.translate_explanation(ai_summary,     SUPPORTED_LANGS.get(response_language, response_language))
        simple_summary = llm.translate_explanation(simple_summary, SUPPORTED_LANGS.get(response_language, response_language))

    print("  [6/6] Building clause risk list + report...")
    clauses: List[ClauseRisk] = []
    raw_clauses = re.split(r"\n{2,}|\n(?=\d+\.|[A-Z][A-Z\s]+:)", en_text)
    for raw in raw_clauses[:15]:
        stripped = raw.strip()
        if len(stripped) < 20:
            continue
        c_risks, _, _, c_raw = run_risk_engine(stripped)
        if c_raw >= 40:
            c_class: RiskClass = "HIGH_RISK"
        elif c_raw >= 15:
            c_class = "MODERATE_RISK"
        else:
            c_class = "LOW_RISK"
        simple, negotiation = llm.explain_clause(stripped[:400])
        clauses.append(ClauseRisk(
            clause      = stripped[:300],
            risk_score  = c_raw,
            risk_class  = c_class,
            explanation = simple,
            negotiation = negotiation,
        ))

    for m in missing:
        clauses.append(ClauseRisk(
            clause      = f"[MISSING] {m}",
            risk_score  = 50,
            risk_class  = "MISSING_CLAUSE",
            explanation = f"This clause is absent: {m}",
            negotiation = f"Insist on adding: {m}",
        ))

    report = {
        "filename":              filename,
        "language":              lang,
        "safety_score":          safety_score,
        "safety_level":          safety_level,
        "risk_count":            len(risks),
        "fraud_signals":         len(fraud),
        "missing_clauses_count": len(missing),
        "top_risks": [
            {"level": r.level, "category": r.category, "suggestion": r.suggestion}
            for r in risks[:5]
        ],
        "fraud_details": [
            {"signal": f.signal, "confidence": f"{f.confidence:.0%}", "detail": f.detail}
            for f in fraud
        ],
        "missing_clauses": missing,
        "ai_summary":      ai_summary,
    }

    return ContractAnalysisResult(
        raw_text        = text,
        language        = lang,
        translated_text = en_text,
        safety_score    = safety_score,
        safety_level    = safety_level,
        risks           = risks,
        clauses         = clauses,
        fraud_signals   = fraud,
        missing_clauses = missing,
        ai_summary      = ai_summary,
        simple_summary  = simple_summary,
        report          = report,
    )


def run_query_pipeline(
    query: str,
    response_language: str = "en",
) -> CaseAssessment:
    print("  [1/4] Detecting language...")
    lang = detect_language(query)

    print(f"  [2/4] Translating query from '{lang}' to English...")
    en_query = translate_text(query, lang, "en") if lang != "en" else query

    print("  [3/4] Retrieving relevant legal context...")
    context = retrieve_context(en_query, top_k=5)

    print("  [4/4] Running AI case assessment...")
    result = llm.assess_case_query(en_query, context)

    reasoning  = result.get("reasoning", "")
    simple_exp = result.get("simple_explanation", "")

    if response_language != "en":
        print(f"  [4b] Translating output to '{response_language}'...")
        reasoning  = llm.translate_explanation(reasoning,  SUPPORTED_LANGS.get(response_language, response_language))
        simple_exp = llm.translate_explanation(simple_exp, SUPPORTED_LANGS.get(response_language, response_language))

    return CaseAssessment(
        query               = query,
        language            = lang,
        translated_query    = en_query,
        relevant_laws       = result.get("relevant_laws", []),
        win_probability     = float(result.get("win_probability", 0.5)),
        risk_level          = result.get("risk_level", "MEDIUM"),
        reasoning           = reasoning,
        recommended_actions = result.get("recommended_actions", []),
        simple_explanation  = simple_exp,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9.  CHAT SESSION
# ─────────────────────────────────────────────────────────────────────────────

class ChatSession:
    def __init__(self, contract_result: Optional[ContractAnalysisResult] = None,
                 language: str = "en"):
        self.history:          List[ChatMessage] = []
        self.language:         str               = language
        self.contract_summary: str               = ""
        if contract_result:
            self.contract_summary = (
                f"Safety: {contract_result.safety_score}/100 ({contract_result.safety_level}). "
                f"Top risks: {', '.join(r.category for r in contract_result.risks[:3])}. "
                f"AI Summary: {contract_result.ai_summary[:300]}"
            )

    def chat(self, user_message: str) -> str:
        msg_lang   = detect_language(user_message)
        en_message = translate_text(user_message, msg_lang, "en") if msg_lang != "en" else user_message
        context    = retrieve_context(en_message, top_k=4)
        answer_en  = llm.chat_answer(en_message, self.history, context, self.contract_summary)
        answer     = answer_en
        if self.language != "en":
            answer = llm.translate_explanation(answer_en, SUPPORTED_LANGS.get(self.language, self.language))
        self.history.append(ChatMessage(role="user",      content=user_message))
        self.history.append(ChatMessage(role="assistant", content=answer))
        return answer

# ─────────────────────────────────────────────────────────────────────────────
# 10.  FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def create_app():
    from fastapi import Body, FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import Optional as Opt

    app = FastAPI(title="KnowRights-AI", version="1.0.0",
                  description="AI Legal Assistant for India")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )

    class QueryRequest(BaseModel):
        query:    str = Field(..., min_length=5)
        language: str = Field("en")

    class ChatRequest(BaseModel):
        message:          str      = Field(..., min_length=1)
        session_id:       str      = Field("default")
        language:         str      = Field("en")
        contract_context: Opt[str] = None

    class ClauseRequest(BaseModel):
        clause: str = Field(..., min_length=5)

    class ContractTextRequest(BaseModel):
        text:     str = Field(..., min_length=10)
        language: str = Field("en")

    _sessions: Dict[str, ChatSession] = {}

    def _contract_response(result: ContractAnalysisResult, fname: str) -> dict:
        return {
            "filename":       fname,
            "language":       result.language,
            "safety_score":   result.safety_score,
            "safety_level":   result.safety_level,
            "risks": [
                {"level": r.level, "category": r.category,
                 "description": r.description, "suggestion": r.suggestion}
                for r in result.risks
            ],
            "fraud_signals": [
                {"signal": f.signal, "confidence": round(f.confidence, 2), "detail": f.detail}
                for f in result.fraud_signals
            ],
            "missing_clauses": result.missing_clauses,
            "clauses": [
                {"clause": c.clause[:200], "risk_score": c.risk_score,
                 "risk_class": c.risk_class, "explanation": c.explanation,
                 "negotiation": c.negotiation}
                for c in result.clauses
            ],
            "ai_summary":     result.ai_summary,
            "simple_summary": result.simple_summary,
            "report":         result.report,
        }

    @app.get("/health")
    def health():
        return {
            "ok":           True,
            "llm_provider": llm.provider or "offline",
            "vectify":      bool(VECTIFY_API_URL),
        }

    def analyze_contract_text(req: ContractTextRequest = Body(...)):
        result = run_contract_pipeline(req.text, filename="text_input",
                                       response_language=req.language)
        return _contract_response(result, "text_input")
    analyze_contract_text.__annotations__["req"] = ContractTextRequest
    app.post("/analyze-contract/text")(analyze_contract_text)

    async def analyze_contract_file(
        file:     UploadFile = File(...),
        language: str        = Form("en"),
    ):
        raw_bytes = await file.read()
        fname     = file.filename or "upload.txt"
        content   = extract_text_from_bytes(raw_bytes, fname)
        result    = run_contract_pipeline(content, filename=fname, response_language=language)
        return _contract_response(result, fname)
    # `from __future__ import annotations` can turn `UploadFile` into a string
    # ForwardRef inside nested functions; explicitly bind it to the real type.
    analyze_contract_file.__annotations__["file"] = UploadFile
    analyze_contract_file.__annotations__["language"] = str
    app.post("/analyze-contract/file")(analyze_contract_file)

    def assess_query(req: QueryRequest = Body(...)):
        a = run_query_pipeline(req.query, response_language=req.language)
        return {
            "query":               a.query,
            "language":            a.language,
            "win_probability":     round(a.win_probability, 2),
            "risk_level":          a.risk_level,
            "relevant_laws":       a.relevant_laws,
            "reasoning":           a.reasoning,
            "recommended_actions": a.recommended_actions,
            "simple_explanation":  a.simple_explanation,
        }

    # `from __future__ import annotations` turns annotations into strings.
    # FastAPI/Pydantic sometimes struggles to resolve the local `QueryRequest`
    # type here, so we explicitly bind the real type object.
    assess_query.__annotations__["req"] = QueryRequest
    app.post("/query")(assess_query)

    def chat_endpoint(req: ChatRequest = Body(...)):
        sid = req.session_id
        if sid not in _sessions:
            _sessions[sid] = ChatSession(language=req.language)
        if req.contract_context:
            _sessions[sid].contract_summary = req.contract_context[:500]
        answer = _sessions[sid].chat(req.message)
        return {"answer": answer, "session_id": sid}
    chat_endpoint.__annotations__["req"] = ChatRequest
    app.post("/chat")(chat_endpoint)

    @app.delete("/chat/{session_id}")
    def clear_session(session_id: str):
        _sessions.pop(session_id, None)
        return {"cleared": session_id}

    def explain_clause(req: ClauseRequest = Body(...)):
        simple, negotiation = llm.explain_clause(req.clause)
        return {"simple_explanation": simple, "negotiation_tip": negotiation}
    explain_clause.__annotations__["req"] = ClauseRequest
    app.post("/explain-clause")(explain_clause)

    @app.get("/languages")
    def list_languages():
        return SUPPORTED_LANGS

    return app

# ─────────────────────────────────────────────────────────────────────────────
# 11.  TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CONTRACT = """
EMPLOYMENT AGREEMENT

This Agreement is entered into between TechCorp Inc. (Company) and the Employee.

1. UNLIMITED LIABILITY: Employee agrees to unlimited liability for any actions taken
   during employment, including indemnifying the company for all claims.

2. TERMINATION: Company may terminate this agreement at any time without cause,
   at its sole discretion, without any notice or severance pay.

3. NON-COMPETE: Employee shall not compete with the Company globally for 5 years
   after termination in any capacity whatsoever.

4. INTELLECTUAL PROPERTY: All intellectual property created by Employee shall
   belong to the Company, including personal projects developed outside working hours.

5. PENALTY: In case of breach, Employee shall forfeit entire salary earned and
   pay a penalty 10 times the annual compensation.

6. AUTO-RENEWAL: This agreement automatically renews yearly unless cancelled
   in writing 90 days in advance.

SIGN IMMEDIATELY - OFFER EXPIRES TODAY.
"""

SAMPLE_FRAUDULENT_CONTRACT = """
INVESTMENT AGREEMENT

Party A: Investor (You)
Party B: Global Wealth Corp Reg No: XXXX (to be decided)

1. No consideration required - gratuitous transfer of funds.
2. No recourse available under any circumstances.
3. Absolute forfeiture of invested amount if any dispute arises.
4. Sign immediately - limited time only offer.
"""

SAMPLE_QUERIES = [
    "If I get into a car crash and it was the other driver's fault, what is my chance of winning a case?",
    "My employer is refusing to pay my salary for 3 months. What can I do?",
    "I received an SMS saying I won a lottery and need to pay Rs 5000 to claim my prize. Is this a scam?",
]


def run_tests():
    print("\n" + "="*70)
    print("  KNOWRIGHTS PIPELINE TEST SUITE")
    print("="*70)

    pass_count = 0
    fail_count = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal pass_count, fail_count
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}]  {name}")
        if detail and not condition:
            print(f"         Detail: {detail}")
        if condition:
            pass_count += 1
        else:
            fail_count += 1

    print("\n-- 1. Language Detection --")
    check("Hindi detected",   detect_language("yeh ek anubandh hai") == "en" or detect_language("यह एक अनुबंध है") == "hi")
    check("English detected", detect_language("This is a contract") == "en")

    print("\n-- 2. Risk Engine --")
    risks, missing, fraud, raw = run_risk_engine(SAMPLE_CONTRACT)
    check("Detects CRITICAL risks",         any(r.level == "CRITICAL" for r in risks))
    check("Detects unlimited liability",    any(r.category == "Unlimited Liability" for r in risks))
    check("Detects non-compete",            any("Non-Compete" in r.category for r in risks))
    check("Detects unilateral termination", any("Termination" in r.category for r in risks))
    check("Detects auto-renewal",           any(r.category == "Auto-Renewal" for r in risks))
    check("Detects missing dispute res.",   any("Dispute" in m for m in missing))
    check("Raw risk score > 50",            raw > 50, f"Got: {raw}")
    print(f"         Risks: {len(risks)}, Missing: {len(missing)}, Raw: {raw}")

    print("\n-- 3. Fraud Detection --")
    _, _, fraud2, _ = run_risk_engine(SAMPLE_FRAUDULENT_CONTRACT)
    check("Detects pressure clause",      any("Pressure"       in f.signal for f in fraud2))
    check("Detects phantom identity",     any("Phantom"        in f.signal for f in fraud2))
    check("Detects unconscionable terms", any("Unconscionable" in f.signal for f in fraud2))

    print("\n-- 4. RL Policy Scoring --")
    safety, label = rl_policy_score(raw, fraud)
    check("Safety score 0-100",                 0 <= safety <= 100)
    check("Extreme/High label for bad contract","EXTREME" in label or "HIGH" in label, f"Got: {label}")
    low_s, low_l = rl_policy_score(0, [])
    check("Safe contract high score",           low_s >= 80, f"Got: {low_s}")

    print("\n-- 5. RAG Retrieval --")
    ctx = retrieve_context("car accident compensation India", top_k=3)
    check("Returns snippets",      len(ctx) >= 1)
    check("Snippets have content", any(len(c) > 20 for c in ctx))

    print("\n-- 6. Full Contract Pipeline --")
    t0      = time.time()
    result  = run_contract_pipeline(SAMPLE_CONTRACT, filename="test.txt")
    elapsed = time.time() - t0
    check("Has safety score",     0 <= result.safety_score <= 100)
    check("Has risks",            len(result.risks) > 0)
    check("Has clauses",          len(result.clauses) > 0)
    check("Has report dict",      isinstance(result.report, dict))
    check("AI summary non-empty", len(result.ai_summary) > 10)
    check("Simple summary",       len(result.simple_summary) > 10)
    print(f"         Time: {elapsed:.1f}s | Safety: {result.safety_score}/100 | {result.safety_level}")

    print("\n-- 7. Fraudulent Contract Pipeline --")
    fr = run_contract_pipeline(SAMPLE_FRAUDULENT_CONTRACT)
    check("Low safety score",       fr.safety_score < 50, f"Got: {fr.safety_score}")
    check("Fraud signals detected", len(fr.fraud_signals) >= 2)

    print("\n-- 8. Query Pipeline --")
    t0         = time.time()
    assessment = run_query_pipeline(SAMPLE_QUERIES[0])
    elapsed    = time.time() - t0
    check("Win prob 0-1",            0.0 <= assessment.win_probability <= 1.0)
    check("Has relevant laws",       len(assessment.relevant_laws) > 0)
    check("Has reasoning",           len(assessment.reasoning) > 10)
    check("Has recommended actions", len(assessment.recommended_actions) > 0)
    check("Has simple explanation",  len(assessment.simple_explanation) > 5)
    print(f"         Time: {elapsed:.1f}s | Win: {assessment.win_probability:.0%} | {assessment.risk_level}")

    print("\n-- 9. Chat Session --")
    session = ChatSession(contract_result=result)
    answer  = session.chat("What are the biggest risks in this contract?")
    check("Chat returns answer", len(answer) > 10)
    check("History recorded",    len(session.history) == 2)

    print("\n-- 10. Clause Explanation --")
    simple, negotiation = llm.explain_clause(
        "Employee shall not compete with the Company globally for 5 years after termination.")
    check("Simple explanation", len(simple) > 5)
    check("Negotiation tip",    len(negotiation) > 5)

    print("\n" + "="*70)
    total = pass_count + fail_count
    print(f"  RESULTS: {pass_count}/{total} passed  |  {fail_count} failed")
    print("  All tests passed!" if fail_count == 0 else "  Some tests failed.")
    print("="*70 + "\n")
    return fail_count == 0

# ─────────────────────────────────────────────────────────────────────────────
# 12.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_cli():
    print("\n" + "="*70)
    print("  KNOWRIGHTS AI - Interactive CLI")
    print("="*70)
    print("\nChoose mode:")
    print("  1 = Analyze a contract")
    print("  2 = Ask a legal query")
    print("  3 = Chat\n")

    mode = input("Enter mode (1/2/3): ").strip()
    lang = input("Language [en/hi/ta/te/bn/mr/gu] (default: en): ").strip() or "en"

    if mode == "1":
        print("\nPaste contract text (type END to finish) or enter file path:")
        line1 = input()
        if Path(line1.strip()).exists():
            with open(line1.strip(), "rb") as fh:
                text = extract_text_from_bytes(fh.read(), line1.strip())
        else:
            lines = [line1]
            while True:
                l = input()
                if l.strip().upper() == "END":
                    break
                lines.append(l)
            text = "\n".join(lines)

        result = run_contract_pipeline(text, response_language=lang)
        print(f"\nSafety: {result.safety_score}/100 | {result.safety_level}")
        print(f"Risks: {len(result.risks)} | Fraud: {len(result.fraud_signals)} | Missing: {len(result.missing_clauses)}")
        print(f"\nAI Summary:\n{result.ai_summary}")
        print(f"\nSimple:\n{result.simple_summary}")

        session = ChatSession(contract_result=result, language=lang)
        while True:
            q = input("\nYou: ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            print(f"\nKnowRights: {session.chat(q)}")

    elif mode == "2":
        query      = input("\nEnter your legal question: ").strip()
        assessment = run_query_pipeline(query, response_language=lang)
        print(f"\nWin Probability : {assessment.win_probability:.0%}")
        print(f"Risk Level      : {assessment.risk_level}")
        print(f"Relevant Laws   : {', '.join(assessment.relevant_laws)}")
        print(f"\nReasoning:\n{assessment.reasoning}")
        print(f"\nSimple:\n{assessment.simple_explanation}")
        print("\nRecommended Actions:")
        for i, a in enumerate(assessment.recommended_actions, 1):
            print(f"  {i}. {a}")

    elif mode == "3":
        session = ChatSession(language=lang)
        while True:
            q = input("\nYou: ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            print(f"\nKnowRights: {session.chat(q)}")

    print("\nThank you for using KnowRights AI!")
    print("Always consult a qualified lawyer for legal decisions.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cli"

    if cmd == "serve":
        import uvicorn
        app  = create_app()
        port = int(os.getenv("PORT", "8000"))
        print(f"\nKnowRights API starting on http://0.0.0.0:{port}")
        print(f"   LLM      : {llm.provider or 'offline'}")
        print(f"   VectifyAI: {'configured' if VECTIFY_API_URL else 'not set (local FAISS)'}")
        print(f"   Docs     : http://0.0.0.0:{port}/docs\n")
        uvicorn.run(app, host="0.0.0.0", port=port)

    elif cmd == "test":
        sys.exit(0 if run_tests() else 1)

    else:
        run_cli()

# Expose ASGI app for production servers (e.g. `uvicorn knowrights_pipeline:app`)
app = create_app()