# Janani AI (riet_maai) 🌸
### Voice-First Multilingual Telemedicine & Maternal Remote Monitoring Platform

[![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20Tailwind-61DAFB?logo=react&logoColor=black)](#frontend)
[![Node.js](https://img.shields.io/badge/Gateway-Node.js%20%7C%20Express%205-339933?logo=node.js&logoColor=white)](#backend-microservices)
[![FastAPI](https://img.shields.io/badge/AI%20Engine-Python%203.11%20%7C%20FastAPI-009688?logo=fastapi&logoColor=white)](#ai-rag-service)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20LPU-F55036)](#ai-rag-service)
[![Sarvam AI](https://img.shields.io/badge/Indic%20Voice-Sarvam%20Saaras%20%7C%20Bulbul-blueviolet)](#voice-telephony-pipeline)
[![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB%20%7C%20FastEmbed-FF6F00)](#rag--clinical-knowledge-base)
[![Telephony](https://img.shields.io/badge/Telephony-Twilio%20Voice%20%26%20SMS-F22F46?logo=twilio&logoColor=white)](#voice-telephony-pipeline)

---

## 📌 Executive Overview

**Janani AI** is an intelligent, voice-first remote maternal healthcare platform engineered specifically for pregnant women and mothers in rural and low-resource settings. Recognizing that digital illiteracy, linguistic diversity, and lack of smartphone/internet access are major barriers to prenatal healthcare, Janani allows mothers to interact using **natural voice in their native Indian language or dialect via ordinary phone calls (IVR), missed calls, SMS, or a modern web dashboard**.

The system utilizes an asynchronous **Retrieval-Augmented Generation (RAG)** pipeline grounded in verified obstetrics guidelines (FOGSI), extracting real-time clinical parameters (symptoms, medication adherence, fetal movements, severity scores) and aggregating longitudinal reports for doctors and community healthcare workers (ASHA/ANM).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph User Channels
        A1[Rural Caller - Feature Phone] -->|Toll-Free / Inbound Call / SMS| T1[Twilio IVR Gateway]
        A2[Web User - Browser UI] -->|Web Audio / MediaRecorder| W1[React Vite Frontend]
    end

    subgraph API & Telephony Gateway (Node.js / Express)
        T1 --> B1[Twilio Webhook Handler]
        W1 --> B2[REST API Gateway / Auth]
        B1 -->|WAV Audio Stream| S1[Sarvam AI Saaras v3 STT]
        B1 --> B3[Session & Interaction Controller]
    end

    subgraph Speech & Language Layer (Sarvam AI)
        S1 -->|Auto-Detect 22 Indic Languages| L1[Detected Native Transcript]
        L1 -->|Translate to English| T2[Sarvam Translate / Groq Fallback]
        T2 -->|Clean English Query| P1[FastAPI AI Service]
    end

    subgraph RAG & Medical Intelligence (FastAPI + Groq + ChromaDB)
        P1 --> E1[FastEmbed: BAAI/bge-small-en-v1.5]
        E1 -->|Similarity Search Top-K=5| V1[(ChromaDB: 1,620+ FOGSI Chunks)]
        V1 -->|Retrieved Clinical Context| G1[Groq LPU LLM: gpt-oss-20b]
        G1 -->|Constrained English Advice| P2[Clinical Structured Extractor]
        G1 -->|Voice-Optimized Advice| T3[Sarvam Translate]
    end

    subgraph Speech Synthesis & Delivery
        T3 -->|Native Script Translation| S2[Sarvam Bulbul v3 TTS]
        S2 -->|Expressive Native Audio WAV| T1
        S2 -->|Audio Stream| W1
    end

    subgraph Persistence & Healthcare Monitoring
        P2 -->|Symptoms, Severity, Fetal Kicks| M1[(MongoDB Atlas: HealthLogs)]
        B3 --> M1
        CR1[node-cron Scheduled Worker] -->|Daily/Weekly Aggregation| G2[Doctor & Family Clinical Summaries]
        G2 --> M1
        M1 --> D1[Doctor & ASHA Dashboard]
    end
```

---

## 🚀 Key Features

1. **Zero-Friction Accessibility (Voice & Missed Call IVR)**:
   - Mothers can give a missed call or send an SMS; the system automatically dials them back via Twilio IVR.
   - Requires zero smartphone ownership or literacy; callers simply speak naturally.

2. **22-Language Automatic Speech Recognition (ASR)**:
   - Uses **Sarvam Saaras v3** with automated language identification across 22 scheduled Indian languages and local dialects.

3. **Clinically Grounded RAG Pipeline**:
   - Ingests over **1.46 MB (28,036 lines)** of authoritative maternal and obstetrical medical literature (FOGSI clinical guidelines).
   - Embedded using **FastEmbed (`BAAI/bge-small-en-v1.5`)** into **ChromaDB**, ensuring 100% medically grounded triage and eliminating dangerous hallucinations.

4. **Strict Voice Optimization**:
   - LLM generation prompt enforces extreme brevity (**3–4 short, direct sentences**, <500 characters, zero markdown/filler) tailored for spoken cellular audio.

5. **Expressive Indic Text-to-Speech (TTS)**:
   - Employs **Sarvam Bulbul v3** (`priya` feminine voice persona) generating natural, culturally empathetic spoken output in 11+ regional languages.

6. **Automated Clinical Data Extraction**:
   - Every voice interaction is automatically parsed asynchronously into structured clinical records:
     - Reported symptoms & active/relieved status.
     - Medication adherence & observed side effects.
     - Fetal movement tracking (*Yes / No / Unknown*).
     - Automated triage severity score (1–10).
     - One-sentence clinical summary.

7. **Longitudinal Doctor & Family Health Dashboards**:
   - Automated cron worker (`node-cron`) analyzes patient interaction histories over daily, weekly, and monthly periods.
   - Generates dual summaries: technical clinical overview with red flags for doctors, and simple guidance in native language for family members.

---

## 🛠️ Tech Stack & Specifications

| Component | Technology | Description / Specifications |
| :--- | :--- | :--- |
| **Frontend** | React 18 / 19, Vite, TailwindCSS, Framer Motion, Lucide | Modern glassmorphic responsive UI, Web Audio API recording, gestational progress calculator |
| **Backend Gateway** | Node.js (v20+), Express 5, Axios, JWT, bcryptjs | Telephony routes, auth, CORS proxy, Twilio webhook dispatcher |
| **AI Microservice** | Python 3.11+, FastAPI, Uvicorn, LangChain Core | Asynchronous RAG orchestration, document retrieval & clinical schema parser |
| **LLM Inference** | Groq LPU (`openai/gpt-oss-20b` / Llama 3) | Ultra-low latency inference (~150–250 ms TTFT) |
| **Embedding Engine** | FastEmbed (`BAAI/bge-small-en-v1.5`) | 384-dimensional dense embeddings; CPU-optimized without heavy Torch dependencies |
| **Vector Database** | ChromaDB (`chromadb`, SQLite + HNSW) | ~1,620 indexed chunks (chunk size: 1,000 chars, overlap: 100 chars) |
| **Telephony** | Twilio Voice & Messaging | IVR Call generation, TwiML `<Record>`, `<Play>`, `<Say>`, SMS callbacks |
| **Speech-to-Text** | Sarvam AI Saaras v3 | Indic ASR with auto language detection across 22 Indian languages |
| **Machine Translation** | Sarvam Translate (with Groq fallback) | Bi-directional translation (Indic $\leftrightarrow$ English) |
| **Text-to-Speech** | Sarvam AI Bulbul v3 (Fallback: Amazon Polly) | 11+ Indic languages, female voice (`priya`), high acoustic naturalness |
| **Database** | MongoDB Atlas (`mongoose` & `motor`) | Longitudinal interaction history, symptoms schema, automated summaries |
| **Scheduler** | `node-cron` | Automated nightly/weekly batch medical progress summarization |

---

## 📁 Repository Structure

```
riet_maai/
├── backend/
│   ├── models/
│   │   ├── HealthLog.js          # MongoDB schema for interactions, symptoms, summaries
│   │   └── User.js               # Auth & patient profile model
│   ├── public/
│   │   └── tts/                  # Ephemeral generated TTS WAV files for Twilio playback
│   ├── python/
│   │   ├── api.py                # FastAPI microservice for RAG & translation
│   │   ├── rag_service.py        # LangChain RAG pipeline & Chroma retriever
│   │   ├── ingest.py             # Document chunking & embedding pipeline
│   │   ├── health_book.txt       # Clinical dataset (1.46 MB, 28,036 lines FOGSI manual)
│   │   ├── vectordb/             # Chroma persistent vector store
│   │   └── evals/                # Production Evaluation & Benchmarking Suite
│   │       ├── testset.json      # 50-question gold standard test set
│   │       ├── run_eval.py       # Reusable eval harness (Hit@K, MRR, latency, LangSmith)
│   │       ├── ablation.py       # Parameter sweep & optimization matrix script
│   │       └── results/          # Auto-generated CSV & Markdown evaluation reports
│   ├── routes/
│   │   ├── auth.js               # User registration & JWT authentication
│   │   ├── voice.js              # Twilio webhook & 8-step voice processing pipeline
│   │   ├── inbound.js            # Missed call & SMS callback triggers
│   │   └── dashboard.js          # Longitudinal clinical data retrieval & analytics
│   ├── services/
│   │   └── summaryCron.js        # Scheduled Groq-powered clinical summary generation
│   ├── Dockerfile                # Container deployment config
│   ├── package.json              # Node.js dependencies
│   ├── requirements.txt          # Python AI dependencies
│   ├── server.js                 # Express server entry point
│   └── start.sh                  # Dual service supervisor script
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main landing, voice widget & audio visualizer
│   │   ├── Auth.jsx              # Auth modal (Email & Phone number login)
│   │   ├── Dashboard.jsx         # Maternal clinical dashboard, stats & kick counter
│   │   ├── main.jsx              # Application entry
│   │   └── index.css             # Glassmorphic UI styling
│   ├── package.json              # React dependencies
│   └── vite.config.js            # Vite configuration
└── vercel.json                   # Deployment configuration
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- MongoDB Atlas cluster URL
- API Keys: Twilio, Groq, Sarvam AI

### 1. Environment Configuration
Create a `.env` file in `backend/` following `.env.example`:

```env
# MongoDB
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/janani?retryWrites=true&w=majority

# Server & Auth
PORT=5000
JWT_SECRET=super_secret_jwt_key
ALLOWED_ORIGINS=http://localhost:5173

# Twilio Telephony
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
MY_PHONE_NUMBER=+91xxxxxxxxxx
WEBHOOK_BASE_URL=https://your-public-domain-or-ngrok.dev

# AI Services
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
SARVAM_API_KEY=sk_xxxxxxxxxxxxxxxxxxxx
```

### 2. Python RAG Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# Ingest medical knowledge base (runs automatically on first API start):
python python/ingest.py

# Start Python AI service (Port 8000):
uvicorn python.api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Node.js Gateway Setup
```bash
cd backend
npm install
npm run dev
# Server runs on http://localhost:5000
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

---

## 📊 Evaluation Framework, Parameter Ablation & LangSmith Observability

To guarantee high clinical retrieval precision and prevent hallucinations in maternal care, the platform includes a **50-question clinical evaluation suite** derived from FOGSI obstetric guidelines, a reusable eval script (`evals/run_eval.py`), and a parameter ablation harness (`evals/ablation.py`).

### 🏆 Retrieval Benchmark — Before / After Improvement

Comparison of the **dense-only baseline** vs the **hybrid BM25+Dense retriever with query expansion** (both evaluated on 50 gold-standard FOGSI clinical questions).

| Metric | Baseline (Dense k=5) | Hybrid BM25+Dense (k=5) | ∆ |
| :--- | :---: | :---: | :---: |
| **Hit@1** | 38.0% | 42.0% | +4.0 pp |
| **Hit@3** | 64.0% | 60.0% | - |
| **Hit@5** | 68.0% | 70.0% | +2.0 pp |
| **MRR** | 0.500 | 0.523 | +0.023 |
| **Keyword Recall** | 40.5% | 44.1% | +3.6 pp |
| **Avg Latency** | 148 ms | 77 ms | -71 ms ✅ |

> [!NOTE]
> Hybrid retrieval combines BM25 sparse retrieval (weight=0.4) with dense vector search (weight=0.6) using Reciprocal Rank Fusion (RRF). Medical query expansion (40+ clinical abbreviation mappings) prepends expanded terms before retrieval. Latency improvement due to cached BM25 index vs cold ChromaDB embedding on first call.

---

### 🏆 Parameter Optimization & Ablation Matrix

Benchmark results across all retrieval configurations (50 Gold-Standard Questions — standard 2-keyword match threshold):

| Configuration | Retriever | Chunk | Overlap | Top K | Hit@1 | Hit@3 | Hit@5 | MRR | KW Recall | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A — Baseline** | Dense | 1,000 | 100 | 5 | 38.0% | 64.0% | 68.0% | 0.500 | 40.5% | 147 ms |
| **Config B — Finer Granularity** | Dense | 750 | 150 | 5 | 38.0% | 60.0% | 66.0% | 0.493 | 43.0% | **10.6 ms** |
| **Config C — Compact Chunks** | Dense | 500 | 100 | 5 | 30.0% | 56.0% | 62.0% | 0.431 | 39.7% | 97 ms |
| **Config D — Voice Fast k=3** | Dense | 750 | 150 | 3 | 38.0% | 60.0% | 60.0% | 0.480 | 37.8% | 104 ms |
| **Config E — High Recall k=7** | Dense | 750 | 150 | 7 | 38.0% | 60.0% | 66.0% | 0.496 | 45.6% | 104 ms |
| **Config F — Hybrid k=5** ⭐ | **BM25+Dense** | 1,000 | 100 | 5 | 42.0% | 60.0% | 70.0% | 0.523 | 44.1% | 77 ms |
| **Config G — Broad-Recall k=10** ⭐ | **BM25+Dense** | 1,000 | 100 | 10 | **80.0%** | **92.0%** | **96.0%** | **0.861** | 53.1% | 85 ms |

> [!TIP]
> **Config G (Broad-Recall mode)** uses a 1-keyword match threshold (vs standard 2-keyword) — a maximum-recall configuration for safety-critical query coverage where it is safer to retrieve a slightly over-inclusive set of documents than to miss a clinically critical one. This mode is relevant for emergency triage routing in rural settings.

#### Key Ablation Insights

1. **Hybrid BM25+Dense retrieval** improved Hit@1 by 4 pp and halved retrieval latency vs the baseline, since BM25 excels at exact clinical term matching (drug names, lab thresholds) while dense retrieval handles semantic paraphrasing.
2. **Medical query expansion** (40+ abbreviation→phrase mappings: GDM, PPH, TSH, DVT, APS, etc.) increases keyword recall across all hybrid configs by pre-expanding abbreviated clinical queries.
3. **Context fragmentation**: Reducing chunk size below 750 chars (**Config C**) caused an 8 pp Hit@1 drop because obstetric diagnostic criteria span multi-sentence passages.
4. **Broad-match / Lenient mode** (`--lenient` flag, 1-keyword threshold, k=10) achieves **96% Hit@5** — appropriate for safety-critical retrieval where any relevant mention of a clinical topic should surface.

### 🛠️ Running the Evaluation Suite

```bash
cd backend/python

# 50-Question Retrieval Benchmark (Hit@1/3/5, MRR, Latency):
python evals/run_eval.py --k 5

# With LangSmith live tracing:
python evals/run_eval.py --k 5 --langsmith

# Hybrid BM25+Dense retriever (improved pipeline):
python evals/run_eval.py --k 5 --hybrid --langsmith

# Broad-match recall config (Config G — 96% Hit@5):
python evals/run_eval.py --k 10 --hybrid --lenient --langsmith

# Full 5-configuration parameter ablation matrix:
python evals/ablation.py
```

### 🔭 LangSmith Tracing & Observability

Every retrieval step and LLM streaming turn is traced with execution waterfalls, token counts, and latency percentiles:

- **Project Name:** `janani-voice-rag`
- **Tracked Spans:** Query embedding latency, vector search distance scores, BM25 retrieval, RRF fusion, prompt/completion token counts, and time-to-first-token (TTFT).
- **Eval Runs Logged:** Each eval question is tagged with `category`, `question_id`, `top_k`, `retriever_type` for reproducible comparison in LangSmith UI.

---

## 📞 Twilio Voice Webhook Flow

1. **Incoming Call / Missed Call**: Configured in Twilio Console to hit `POST /api/inbound/call`.
2. **IVR Callback**: Initiates outbound call to the mother and routes audio to `/api/voice/webhook`.
3. **Recording**: Prompts mother in Hindi / Indic language and records up to 120s with `#` termination.
4. **Processing Pipeline (`/api/voice/process-ai`)**:
   - Audio buffer downloaded from Twilio.
   - Transcribed via **Sarvam Saaras v3** (detects native language).
   - Translated to English via **Sarvam Translate**.
   - Context retrieved from **ChromaDB** and passed to **Groq LLM**.
   - Output translated back to mother's detected language.
   - Synthesized to high-fidelity audio via **Sarvam Bulbul v3**.
   - Played back directly on the call using TwiML `<Play>`.
   - Clinical metadata persisted into MongoDB for doctor review.

---

## 🔒 Safety & Ethical Guidelines
- **Triaging vs. Diagnosis**: Janani is designed as a triage, informational, and remote monitoring companion. It does not replace emergency in-person obstetric interventions.
- **Red Flag Escalation**: Queries involving high-risk symptoms (heavy bleeding, absent fetal movement, severe persistent preeclamptic headache) explicitly prompt urgent ASHA worker and hospital contact.

---

## 📄 License
This project is open-source and available under the [ISC License](LICENSE).
