"""
Janani AI - Reusable Evaluation Harness for Maternal Health RAG Pipeline
Evaluates Retrieval Quality (Hit@K, MRR), Answer Quality, Latency Profiling, and LangSmith Logging.
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Ensure python path can find sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Load environment
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

# Set LangSmith defaults if key provided
if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = "janani-rag-eval"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"


def check_chunk_relevance(chunk_text: str, gold_keywords: List[str], lenient: bool = False) -> bool:
    """Checks if a retrieved chunk matches the gold source criteria.
    
    Args:
        chunk_text: Retrieved document text.
        gold_keywords: Expected keywords for this question.
        lenient: If True, only 1 keyword match required (broad-match / max-recall mode).
    """
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in gold_keywords if kw.lower() in text_lower)
    if lenient:
        threshold = 1
    else:
        # Standard: at least 2 keywords match, or >= 50% of keywords
        threshold = max(1, min(2, len(gold_keywords) // 2))
    return matches >= threshold


def compute_retrieval_metrics(retrieved_docs: List[Any], gold_keywords: List[str], k_values=(1, 3, 5), lenient: bool = False) -> Dict[str, Any]:
    """Computes Hit@K, reciprocal rank, and keyword coverage."""
    first_hit_rank = 0
    hit_at_k = {f"hit@{k}": 0 for k in k_values}
    relevant_chunks = 0

    all_retrieved_text = " ".join([getattr(d, "page_content", str(d)) for d in retrieved_docs]).lower()
    matched_kws = [kw for kw in gold_keywords if kw.lower() in all_retrieved_text]
    keyword_coverage = len(matched_kws) / max(len(gold_keywords), 1)

    for rank, doc in enumerate(retrieved_docs, start=1):
        content = getattr(doc, "page_content", str(doc))
        if check_chunk_relevance(content, gold_keywords, lenient=lenient):
            relevant_chunks += 1
            if first_hit_rank == 0:
                first_hit_rank = rank

    for k in k_values:
        hit_at_k[f"hit@{k}"] = 1 if (first_hit_rank > 0 and first_hit_rank <= k) else 0

    reciprocal_rank = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0
    precision_at_k = relevant_chunks / max(len(retrieved_docs), 1)

    return {
        **hit_at_k,
        "mrr": reciprocal_rank,
        "first_hit_rank": first_hit_rank,
        "precision": precision_at_k,
        "keyword_coverage": keyword_coverage,
    }


def evaluate_answer_constraints(answer: str) -> Dict[str, Any]:
    """Evaluates strict voice constraints for rural telephone delivery."""
    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]
    sentence_count = len(sentences)
    char_count = len(answer)
    has_markdown = any(char in answer for char in ["*", "#", "`", "[", "]", "- "])
    is_brief = sentence_count <= 4 and char_count <= 550
    passes_voice = is_brief and (not has_markdown)

    return {
        "sentence_count": sentence_count,
        "char_count": char_count,
        "has_markdown": has_markdown,
        "passes_voice_constraint": passes_voice,
    }


def compute_ground_truth_similarity(generated: str, expected: str) -> float:
    """Computes token-level F1 overlap between generated answer and expected answer."""
    gen_tokens = set(generated.lower().replace(".", " ").replace(",", " ").split())
    exp_tokens = set(expected.lower().replace(".", " ").replace(",", " ").split())
    if not gen_tokens or not exp_tokens:
        return 0.0
    intersection = gen_tokens.intersection(exp_tokens)
    if not intersection:
        return 0.0
    precision = len(intersection) / len(gen_tokens)
    recall = len(intersection) / len(exp_tokens)
    return (2 * precision * recall) / (precision + recall)


def run_evaluation(
    testset_path: str,
    top_k: int = 5,
    retrieval_only: bool = True,
    persist_directory: str = "vectordb",
    output_dir: Optional[str] = None,
    log_to_langsmith: bool = False,
    use_hybrid: bool = False,
    lenient: bool = False,
) -> Dict[str, Any]:
    """Runs full evaluation benchmark across all test cases."""
    retriever_label = "Hybrid BM25+Dense" if use_hybrid else "Dense (baseline)"
    match_mode = "Lenient (threshold=1 keyword)" if lenient else "Standard (threshold=2 keywords)"
    print(f"\n=======================================================")
    print(f"🏥 Janani AI - RAG Evaluation Harness")
    print(f"=======================================================")
    print(f"📂 Testset:          {testset_path}")
    print(f"🎯 Top K:            {top_k}")
    print(f"⚡ Retrieval Only:   {retrieval_only}")
    print(f"🔍 Retriever:        {retriever_label}")
    print(f"🎚️  Match Mode:       {match_mode}")
    print(f"💾 Vector DB:        {persist_directory}")
    print(f"📊 LangSmith Log:    {log_to_langsmith and bool(os.getenv('LANGCHAIN_API_KEY'))}")
    print(f"=======================================================\n")

    # Load testset
    with open(testset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize retriever
    base_dir = Path(__file__).resolve().parent.parent
    full_persist_dir = str(base_dir / persist_directory)

    if use_hybrid:
        # Hybrid BM25 + Dense retriever with query expansion
        sys.path.insert(0, str(base_dir))
        from hybrid_rag_service import HybridPregnancyRAGService
        hybrid_svc = HybridPregnancyRAGService(persist_directory=persist_directory, k=top_k)
        retriever = hybrid_svc
    else:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        from langchain_chroma import Chroma
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        if not os.path.exists(full_persist_dir):
            print(f"⚠️ Vector DB not found at {full_persist_dir}. Running ingest from health_book.txt...")
            from ingest import ingest_docs
            health_file = str(base_dir / "health_book.txt")
            ingest_docs(health_file, full_persist_dir)

        vectordb = Chroma(
            persist_directory=full_persist_dir,
            embedding_function=embeddings,
            collection_name="pregnancy_docs"
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    # Optional LangSmith setup
    ls_client = None
    if log_to_langsmith and os.getenv("LANGCHAIN_API_KEY"):
        try:
            from langsmith import Client
            ls_client = Client()
            print(" Connected to LangSmith for experiment logging.")
        except Exception as e:
            print(f"⚠️ LangSmith connection notice: {e}")

    results = []
    retrieval_latencies = []
    generation_latencies = []

    print(f"Running evaluation on {len(test_cases)} clinical questions...\n")

    for i, test in enumerate(test_cases, start=1):
        q_id = test["id"]
        category = test.get("category", "general")
        question = test["question"]
        gold_keywords = test["gold_keywords"]
        expected_answer = test["expected_answer"]

        # 1. Measure Retrieval with LangSmith Tracing
        t0 = time.perf_counter()
        config = None
        if log_to_langsmith or os.getenv("LANGCHAIN_TRACING_V2") == "true":
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(
                run_name=f"Eval_Q{q_id:02d}_{category}",
                tags=["eval_harness", category, f"top_k_{top_k}"],
                metadata={"question_id": q_id, "category": category, "expected_answer": expected_answer}
            )
        
        retrieved_docs = retriever.invoke(question, config=config) if config else retriever.invoke(question)
        t_retrieval = (time.perf_counter() - t0) * 1000  # ms
        retrieval_latencies.append(t_retrieval)

        ret_metrics = compute_retrieval_metrics(retrieved_docs, gold_keywords, k_values=(1, 3, 5), lenient=lenient)

        # 2. Answer generation (optional)
        ans_metrics = {}
        t_gen = 0.0
        generated_answer = ""
        
        if not retrieval_only and os.getenv("GROQ_API_KEY"):
            from rag_service import PregnancyRAGService
            rag = PregnancyRAGService(persist_directory=full_persist_dir)
            t_gen_start = time.perf_counter()
            stream = rag.ask_stream(question, "Evaluator Test")
            generated_answer = "".join(list(stream))
            t_gen = (time.perf_counter() - t_gen_start) * 1000
            generation_latencies.append(t_gen)

            ans_metrics = evaluate_answer_constraints(generated_answer)
            ans_metrics["ground_truth_f1"] = compute_ground_truth_similarity(generated_answer, expected_answer)

        record = {
            "id": q_id,
            "category": category,
            "question": question,
            "retrieval_ms": round(t_retrieval, 2),
            **ret_metrics,
            **ans_metrics
        }
        results.append(record)

        hit_indicator = "✅ Hit@1" if ret_metrics["hit@1"] else ("🟢 Hit@5" if ret_metrics["hit@5"] else "❌ Miss")
        print(f"[{i:02d}/{len(test_cases):02d}] ({category[:14]:<14}) {hit_indicator:<8} | MRR: {ret_metrics['mrr']:.2f} | Latency: {t_retrieval:.1f}ms | '{question[:45]}...'")

    # Aggregate Metrics
    n = len(results)
    avg_hit1 = (sum(r["hit@1"] for r in results) / n) * 100
    avg_hit3 = (sum(r["hit@3"] for r in results) / n) * 100
    avg_hit5 = (sum(r["hit@5"] for r in results) / n) * 100
    avg_mrr = sum(r["mrr"] for r in results) / n
    avg_precision = sum(r["precision"] for r in results) / n
    avg_kw_cov = sum(r["keyword_coverage"] for r in results) / n

    p50_latency = statistics.median(retrieval_latencies)
    p90_latency = statistics.quantiles(retrieval_latencies, n=10)[8] if len(retrieval_latencies) >= 10 else max(retrieval_latencies)
    p95_latency = statistics.quantiles(retrieval_latencies, n=20)[18] if len(retrieval_latencies) >= 20 else max(retrieval_latencies)
    avg_latency = statistics.mean(retrieval_latencies)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": n,
        "top_k": top_k,
        "retriever": "hybrid_bm25_dense" if use_hybrid else "dense_baseline",
        "match_mode": "lenient" if lenient else "standard",
        "hit@1_percent": round(avg_hit1, 2),
        "hit@3_percent": round(avg_hit3, 2),
        "hit@5_percent": round(avg_hit5, 2),
        "mean_reciprocal_rank_mrr": round(avg_mrr, 4),
        "precision_at_k": round(avg_precision, 4),
        "keyword_coverage_percent": round(avg_kw_cov * 100, 2),
        "retrieval_latency_ms": {
            "avg": round(avg_latency, 2),
            "p50": round(p50_latency, 2),
            "p90": round(p90_latency, 2),
            "p95": round(p95_latency, 2)
        }
    }

    print("\n" + "="*55)
    print("🎯 AGGREGATE EVALUATION REPORT")
    print("="*55)
    print(f"Total Test Cases:       {n}")
    print(f"Retriever:              {summary['retriever']}")
    print(f"Match Mode:             {summary['match_mode']}")
    print(f"Hit@1 Accuracy:         {summary['hit@1_percent']}%")
    print(f"Hit@3 Accuracy:         {summary['hit@3_percent']}%")
    print(f"Hit@5 Accuracy:         {summary['hit@5_percent']}%")
    print(f"Mean Reciprocal Rank:   {summary['mean_reciprocal_rank_mrr']}")
    print(f"Keyword Recall:         {summary['keyword_coverage_percent']}%")
    print(f"Avg Retrieval Latency:  {summary['retrieval_latency_ms']['avg']} ms (p50: {p50_latency:.1f}ms, p95: {p95_latency:.1f}ms)")
    print("="*55 + "\n")

    # Save to disk
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = out_p / f"eval_report_{timestamp_str}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": results}, f, indent=2)

        # Save Markdown Summary
        md_file = out_p / f"eval_summary_{timestamp_str}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Janani AI - RAG Evaluation Report\n")
            f.write(f"**Generated:** {summary['timestamp']} | **Top-K:** {top_k}\n\n")
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| **Test Questions** | {n} |\n")
            f.write(f"| **Hit@1** | {summary['hit@1_percent']}% |\n")
            f.write(f"| **Hit@3** | {summary['hit@3_percent']}% |\n")
            f.write(f"| **Hit@5** | {summary['hit@5_percent']}% |\n")
            f.write(f"| **MRR (Mean Reciprocal Rank)** | {summary['mean_reciprocal_rank_mrr']} |\n")
            f.write(f"| **Keyword Coverage** | {summary['keyword_coverage_percent']}% |\n")
            f.write(f"| **Avg Retrieval Latency** | {summary['retrieval_latency_ms']['avg']} ms |\n")
            f.write(f"| **p95 Latency** | {summary['retrieval_latency_ms']['p95']} ms |\n")

        print(f"📁 Reports saved to:\n  - {json_file}\n  - {md_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janani AI Evaluation Harness")
    parser.add_argument("--testset", type=str, default=str(Path(__file__).parent / "testset.json"))
    parser.add_argument("--k", type=int, default=5, help="Top K documents to retrieve")
    parser.add_argument("--full", action="store_true", help="Run full RAG generation evaluation if Groq key exists")
    parser.add_argument("--output", type=str, default=str(Path(__file__).parent / "results"))
    parser.add_argument("--langsmith", action="store_true", help="Log runs to LangSmith")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid BM25+dense retriever with query expansion")
    parser.add_argument("--lenient", action="store_true", help="Lenient keyword match (threshold=1); broad-recall config")
    parser.add_argument("--persist-dir", type=str, default="vectordb", help="ChromaDB persist directory (relative to python/)")

    args = parser.parse_args()
    run_evaluation(
        testset_path=args.testset,
        top_k=args.k,
        retrieval_only=not args.full,
        persist_directory=args.persist_dir,
        output_dir=args.output,
        log_to_langsmith=args.langsmith,
        use_hybrid=args.hybrid,
        lenient=args.lenient,
    )
