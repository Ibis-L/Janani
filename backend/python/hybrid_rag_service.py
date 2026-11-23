"""
Janani AI - Hybrid Retrieval Service
Combines BM25 sparse retrieval + FastEmbed dense retrieval via Reciprocal Rank Fusion (RRF),
with medical query expansion for improved recall on clinical terminology.

Key capabilities:
  1. BM25Retriever  — exact keyword match, optimized for clinical abbreviations
  2. Dense Retriever — semantic vector similarity using BAAI/bge-small-en-v1.5
  3. Weighted Reciprocal Rank Fusion (RRF) — robust ensemble without external dependencies
  4. Medical query expansion — maps 40+ obstetrical abbreviations before retrieval
  5. Full drop-in parity with PregnancyRAGService (supports ask_stream, ask, get_context_and_sources)
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

# Ensure parent directory is in sys.path so sibling modules (ingest, etc.) always resolve
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from dotenv import load_dotenv
_env_path = _current_dir.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

# LangSmith Auto-configuration
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "janani-voice-rag")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

# ---------------------------------------------------------------------------
# Medical Query Expansion Dictionary
# Maps common abbreviations and short forms → expanded clinical phrases
# This dramatically improves BM25 recall for acronym-heavy clinical notes.
# ---------------------------------------------------------------------------
MEDICAL_EXPANSIONS: Dict[str, str] = {
    # Obstetric / Pregnancy
    "gdm":    "gestational diabetes mellitus GDM glucose insulin pregnancy",
    "pph":    "postpartum hemorrhage PPH bleeding uterine atony oxytocin",
    "pih":    "pregnancy-induced hypertension PIH blood pressure preeclampsia",
    "pe":     "preeclampsia eclampsia hypertension proteinuria seizure",
    "hellp":  "HELLP syndrome hemolysis elevated liver enzymes low platelet preeclampsia",
    "iugr":   "intrauterine growth restriction IUGR fetal growth small for gestational age",
    "pprom":  "preterm premature rupture of membranes PPROM water break preterm labor",
    "rom":    "rupture of membranes ROM water breaking amniotic fluid",
    "ctx":    "contractions uterine contractions labor pains cervical dilation",
    "cs":     "caesarean section C-section surgical delivery",
    "vbac":   "vaginal birth after caesarean VBAC uterine rupture trial of labor",
    "apgar":  "APGAR score newborn assessment birth heart rate breathing",
    "bmi":    "body mass index BMI weight obesity overweight pregnancy",
    # Anaemia
    "hb":     "haemoglobin hemoglobin Hb anaemia iron deficiency blood",
    "ida":    "iron deficiency anaemia IDA ferritin iron supplementation",
    "ifs":    "iron folate supplementation IFA anaemia prevention pregnancy",
    # Thyroid
    "tsh":    "thyroid stimulating hormone TSH hypothyroidism hyperthyroidism",
    "t4":     "thyroxine T4 thyroid hormone levothyroxine",
    # Hepatitis / TB
    "hbsag":  "hepatitis B surface antigen HBsAg viral hepatitis liver disease",
    "hcv":    "hepatitis C virus HCV liver disease jaundice",
    "tb":     "tuberculosis TB mycobacterium anti-tubercular treatment isoniazid rifampin",
    "att":    "anti-tubercular therapy ATT TB treatment isoniazid rifampin pyrazinamide",
    # APS / Thromboembolic
    "aps":    "antiphospholipid syndrome APS thrombosis miscarriage aspirin heparin",
    "dvt":    "deep vein thrombosis DVT clot anticoagulant heparin",
    "pe_vte": "pulmonary embolism VTE thromboembolism dyspnoea heparin",
    "lmwh":   "low molecular weight heparin LMWH enoxaparin thrombosis",
    # Cardiac
    "ms":     "mitral stenosis MS cardiac valve disease rheumatic heart disease",
    "af":     "atrial fibrillation AF arrhythmia cardiac",
    # UTI
    "uti":    "urinary tract infection UTI dysuria bacteriuria nitrofurantoin",
    "asb":    "asymptomatic bacteriuria ASB urine culture antibiotic",
    # Fetal Monitoring
    "nst":    "non-stress test NST fetal heart rate monitoring biophysical profile",
    "bpp":    "biophysical profile BPP fetal well-being amniotic fluid",
    "afp":    "alpha-fetoprotein AFP maternal serum screening neural tube defect",
    "ctg":    "cardiotocograph CTG fetal heart rate monitoring contractions",
    # Emergency
    "mews":   "modified early obstetric warning score MEWS critical care deterioration",
    "icu":    "intensive care unit ICU critical obstetric care mechanically ventilated",
    "dka":    "diabetic ketoacidosis DKA hyperglycemia insulin emergency",
    # General
    "bp":     "blood pressure BP hypertension systolic diastolic",
    "hr":     "heart rate HR pulse tachycardia bradycardia",
    "rr":     "respiratory rate RR breathing dyspnoea tachypnoea",
    "spo2":   "oxygen saturation SpO2 pulse oximetry hypoxia",
    "iv":     "intravenous IV fluid drip infusion",
    "im":     "intramuscular IM injection",
}


def expand_query(query: str) -> str:
    """
    Expands medical abbreviations in a query to improve retrieval recall.
    Original query terms are preserved; expansions are appended.
    """
    expanded_terms = []
    tokens = re.findall(r'\b\w+\b', query.lower())
    seen = set()
    for token in tokens:
        if token in MEDICAL_EXPANSIONS and token not in seen:
            expanded_terms.append(MEDICAL_EXPANSIONS[token])
            seen.add(token)
    if expanded_terms:
        return query + " " + " ".join(expanded_terms)
    return query


class HybridPregnancyRAGService:
    """
    Hybrid retrieval service: BM25 sparse + Dense vector ensemble with query expansion.
    Drop-in replacement for PregnancyRAGService — supports .invoke(), .ask_stream(), and .ask().
    """

    def __init__(
        self,
        persist_directory: str = "vectordb",
        k: int = 5,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        self.k = k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # 1. Initialize LLM (Groq Llama-3 / gpt-oss-20b)
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="openai/gpt-oss-20b",
            groq_api_key=groq_api_key
        ) if groq_api_key else None

        # 2. Initialize Vector DB
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        base_dir = Path(__file__).resolve().parent
        full_persist_dir = str(base_dir / persist_directory)

        if not os.path.exists(full_persist_dir) or not any(Path(full_persist_dir).iterdir()):
            print(f"⚠️ VectorDB at {persist_directory} not found. Ingesting from health_book.txt...")
            health_file = base_dir / "health_book.txt"
            if health_file.exists():
                from ingest import ingest_docs
                ingest_docs(str(health_file), full_persist_dir)

        self.vectordb = Chroma(
            persist_directory=full_persist_dir,
            embedding_function=embeddings,
            collection_name="pregnancy_docs",
        )

        fetch_k = max(k * 2, 10)
        self._dense = self.vectordb.as_retriever(search_kwargs={"k": fetch_k})

        # 3. Build BM25 Retriever from ChromaDB document corpus
        try:
            collection = self.vectordb._collection
            raw = collection.get(include=["documents", "metadatas"])
            docs_text = raw.get("documents", []) or []
            metadatas = raw.get("metadatas", []) or []
        except Exception:
            docs_text = []
            metadatas = []

        if docs_text:
            all_docs = [
                Document(page_content=txt, metadata=meta or {})
                for txt, meta in zip(docs_text, metadatas)
            ]
            self._bm25 = BM25Retriever.from_documents(all_docs)
            self._bm25.k = fetch_k
            self._has_bm25 = True
            print(f"✅ Hybrid retriever ready (BM25 w={bm25_weight} + Dense w={dense_weight}, k={k})")
        else:
            self._bm25 = None
            self._has_bm25 = False
            print("⚠️ BM25 corpus empty — operating in dense-only mode")

        # 4. Standard voice prompt definition
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert prenatal care evaluator for pregnant women. You will receive transcribed audio input regarding a woman's current symptoms, diet, and lifestyle habits.
Your sole purpose is to evaluate this information and provide immediate, practical guidance.

CRITICAL CONSTRAINTS:
- EXTREME BREVITY: Strict token limit. Keep your response under 3 to 4 short sentences.
- VOICE-OPTIMIZED: Your output will be spoken directly to the user via Text-to-Speech. Speak in a warm, simple, and direct conversational tone. Do absolutely NOT use markdown, asterisks, bullet points, or special characters.
- NO FLUFF: Do not use conversational filler. Get straight to the solution.
- DIRECT INFORMATION: Provide direct actionable information without disclaimers.

RESPONSE STRUCTURE:
1. Symptom/Emergency: If she reports a symptom, give the immediate action to take right now.
2. Diet: Give one specific, easily accessible food addition or subtraction based on her input.
3. Lifestyle: Give one specific adjustment for her daily routine or prenatal care.

Be impactful, highly specific, and concise."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """CONTEXT FROM MEDICAL DOCUMENTS:
{context}

PATIENT DATA:
{patient_data}

USER QUESTION:
{question}

JANANI RESPONSE:""")
        ])

        # Expose retriever interface for LangChain compatibility
        self.retriever = self

    def _rrf_merge(self, lists: List[List[Any]], weights: List[float], k_rrf: int = 60) -> List[Any]:
        """Weighted Reciprocal Rank Fusion (RRF) across ranked lists."""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}
        for doc_list, w in zip(lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                key = getattr(doc, "page_content", str(doc))[:200]
                rrf_score = w / (k_rrf + rank)
                scores[key] = scores.get(key, 0.0) + rrf_score
                doc_map[key] = doc
        ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[k] for k in ranked_keys]

    def invoke(self, query: str, config=None) -> List[Any]:
        """
        Retrieves top-k documents using medical query expansion + RRF ensemble.
        """
        expanded = expand_query(query)

        if self._has_bm25 and self._bm25 is not None:
            bm25_docs = self._bm25.invoke(expanded) if config is None else self._bm25.invoke(expanded, config=config)
            dense_docs = self._dense.invoke(expanded) if config is None else self._dense.invoke(expanded, config=config)
            docs = self._rrf_merge([bm25_docs, dense_docs], [self.bm25_weight, self.dense_weight])
        else:
            docs = self._dense.invoke(expanded) if config is None else self._dense.invoke(expanded, config=config)

        # Deduplicate
        seen_content = set()
        unique_docs = []
        for d in docs:
            h = hash(getattr(d, "page_content", str(d))[:200])
            if h not in seen_content:
                seen_content.add(h)
                unique_docs.append(d)

        return unique_docs[: self.k]

    def ask_stream(self, query: str, patient_data: str = "None provided", chat_history: list = None) -> Generator[str, None, None]:
        """Streaming response generator with LangSmith tracing."""
        if chat_history is None:
            chat_history = []

        if self.llm is None:
            yield "LLM API key not configured. Please set GROQ_API_KEY in .env."
            return

        docs = self.invoke(
            query,
            config=RunnableConfig(
                run_name="Janani_Hybrid_Retrieval",
                tags=["hybrid_retriever", "bm25", "fastembed", "rrf"]
            )
        )
        context = "\n\n".join([d.page_content for d in docs])

        generation_chain = self.rag_prompt | self.llm | StrOutputParser()
        run_config = RunnableConfig(
            run_name="Janani_Hybrid_Voice_Turn",
            tags=["hybrid_voice_rag", "maternal_care", "llama3_groq"],
            metadata={
                "patient_data_length": len(patient_data),
                "retrieved_doc_count": len(docs)
            }
        )

        for chunk in generation_chain.stream(
            {
                "chat_history": chat_history,
                "context": context,
                "question": query,
                "patient_data": patient_data
            },
            config=run_config
        ):
            clean_chunk = chunk.replace("*", "").replace("#", "").replace("- ", "")
            yield clean_chunk

    def ask(self, query: str, patient_data: str = "None provided", chat_history: list = None) -> Dict[str, Any]:
        """Non-streaming query for synchronous evaluation and debugging."""
        chunks = list(self.ask_stream(query, patient_data, chat_history))
        answer = "".join(chunks).strip()
        context, sources = self.get_context_and_sources(query)
        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }

    def get_context_and_sources(self, query: str):
        docs = self.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
        return context, sources


if __name__ == "__main__":
    svc = HybridPregnancyRAGService(k=5)
    q = "What are the management steps for PPH?"
    expanded = expand_query(q)
    print(f"\nOriginal  : {q}")
    print(f"Expanded  : {expanded}\n")
    docs = svc.invoke(q)
    print(f"Retrieved {len(docs)} documents successfully.")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.page_content[:100]}...")
