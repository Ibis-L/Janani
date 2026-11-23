import os
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from pathlib import Path
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

# LangSmith Auto-configuration
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "janani-voice-rag")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

class PregnancyRAGService:
    def __init__(self, persist_directory: str = "vectordb"):
        # 1. Initialize LLM (Llama 3 via Groq)
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="openai/gpt-oss-20b",  # Fast model for voice RAG
            groq_api_key=groq_api_key
        ) if groq_api_key else None

        # 2. Initialize Vector DB
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Ensure vectordb exists and is populated from health_book.txt
        index_exists = os.path.exists(persist_directory) and any(os.listdir(persist_directory))
        if not index_exists:
            print(f"⚠️ VectorDB at {persist_directory} not found or empty. Initializing from health_book.txt...")
            from ingest import ingest_docs
            health_file = Path(__file__).resolve().parent / "health_book.txt"
            if health_file.exists():
                ingest_docs(str(health_file), persist_directory)
            else:
                print(f"❌ Error: {health_file} not found. RAG service will have no medical context.")

        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="pregnancy_docs"
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})

        # 3. Prompt - Expert Prenatal Care Evaluator
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert prenatal care evaluator for pregnant women. You will receive transcribed audio input regarding a woman's current symptoms, diet, and lifestyle habits.
Your sole purpose is to evaluate this information and provide immediate, practical guidance.

CRITICAL CONSTRAINTS:
- EXTREME BREVITY: Strict token limit. Keep your response under 3 to 4 short sentences.
- VOICE-OPTIMIZED: Your output will be spoken directly to the user via Text-to-Speech. Speak in a warm, simple, and direct conversational tone. Do absolutely NOT use markdown, asterisks, bullet points, or special characters.
- NO FLUFF: Do not use conversational filler (e.g., 'Thank you for sharing', 'I understand'). Get straight to the solution.
- DIRECT INFORMATION: Provide direct actionable information. Do not include disclaimers.

RESPONSE STRUCTURE (Answer only what is relevant):
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

    def ask_stream(self, query: str, patient_data: str = "None provided", chat_history: list = None) -> Generator[str, None, None]:
        if chat_history is None:
            chat_history = []
            
        if self.llm is None:
            yield "LLM API key not configured. Please set GROQ_API_KEY in .env."
            return

        # 1. Retrieve
        docs = self.retriever.invoke(
            query,
            config=RunnableConfig(
                run_name="Janani_Doc_Retrieval",
                tags=["retriever", "fastembed", "chromadb"]
            )
        )
        context = "\n\n".join([d.page_content for d in docs])
        
        # 2. Streaming Generation with LangSmith Tracing
        generation_chain = self.rag_prompt | self.llm | StrOutputParser()
        
        run_config = RunnableConfig(
            run_name="Janani_Voice_RAG_Turn",
            tags=["voice_rag", "maternal_care", "llama3_groq"],
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
            # Strip asterisks and other markdown on the fly
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
        docs = self.retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
        return context, sources

if __name__ == "__main__":
    service = PregnancyRAGService()
    docs = service.retriever.invoke("What should I do about morning sickness?")
    print(f"Retrieved {len(docs)} documents successfully.")
