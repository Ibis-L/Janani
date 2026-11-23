"""
Janani AI - Parameter Ablation & Optimization Suite
Tests RAG performance across different chunk sizes, overlaps, and top_k retrieval depths.
Outputs a structured Before-vs-After comparison matrix for documentation and resume citations.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Ensure python path can find sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from run_eval import compute_retrieval_metrics


def manual_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks


def build_ablation_index(
    text_path: str,
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings: Any
) -> Chroma:
    """Builds or returns a Chroma index configured for a specific chunk geometry."""
    if os.path.exists(persist_dir) and any(os.listdir(persist_dir)):
        # Already indexed
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="pregnancy_docs"
        )

    print(f"🔨 Building index: chunk_size={chunk_size}, overlap={chunk_overlap} -> {persist_dir}")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = manual_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = [Document(page_content=c, metadata={"source": text_path, "chunk_id": idx}) for idx, c in enumerate(chunks)]
    print(f"   Indexed {len(docs)} chunks.")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="pregnancy_docs"
    )
    return vectordb


def run_ablation_experiment(
    testset_path: str,
    health_book_path: str,
    output_dir: str
):
    print("\n" + "="*70)
    print("🔬 Janani AI - RAG Parameter Ablation & Optimization Matrix")
    print("="*70)

    with open(testset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    base_eval_dir = Path(__file__).resolve().parent

    # Define the parameter configurations (including Hybrid and Broad-Recall configs)
    configs = [
        {
            "name": "Config A (Baseline)",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "top_k": 5,
            "retriever_type": "dense",
            "lenient": False,
            "desc": "Original repo baseline"
        },
        {
            "name": "Config B (Finer Granularity)",
            "chunk_size": 750,
            "chunk_overlap": 150,
            "top_k": 5,
            "retriever_type": "dense",
            "lenient": False,
            "desc": "Balanced clinical chunking"
        },
        {
            "name": "Config C (Compact Chunks)",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "top_k": 5,
            "retriever_type": "dense",
            "lenient": False,
            "desc": "High specificity, compact"
        },
        {
            "name": "Config D (Fast Voice Low-K)",
            "chunk_size": 750,
            "chunk_overlap": 150,
            "top_k": 3,
            "retriever_type": "dense",
            "lenient": False,
            "desc": "Ultra-low token footprint for phone calls"
        },
        {
            "name": "Config E (High Recall Depth)",
            "chunk_size": 750,
            "chunk_overlap": 150,
            "top_k": 7,
            "retriever_type": "dense",
            "lenient": False,
            "desc": "Deep clinical context recall"
        },
        {
            "name": "Config F (Hybrid BM25+Dense)",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "top_k": 5,
            "retriever_type": "hybrid",
            "lenient": False,
            "desc": "BM25 + dense ensemble with medical query expansion"
        },
        {
            "name": "Config G (Broad-Recall k=10)",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "top_k": 10,
            "retriever_type": "hybrid",
            "lenient": True,
            "desc": "Safety-critical maximum recall mode (1-kw threshold)"
        }
    ]

    summary_rows = []

    for cfg in configs:
        cfg_name = cfg["name"]
        c_size = cfg["chunk_size"]
        c_over = cfg["chunk_overlap"]
        top_k = cfg["top_k"]
        ret_type = cfg.get("retriever_type", "dense")
        is_lenient = cfg.get("lenient", False)

        idx_dir = str(base_eval_dir / "indices" / f"index_cs{c_size}_co{c_over}")

        if ret_type == "hybrid":
            from hybrid_rag_service import HybridPregnancyRAGService
            rel_idx_dir = f"evals/indices/index_cs{c_size}_co{c_over}"
            retriever = HybridPregnancyRAGService(persist_directory=rel_idx_dir, k=top_k)
        else:
            vectordb = build_ablation_index(
                text_path=health_book_path,
                persist_dir=idx_dir,
                chunk_size=c_size,
                chunk_overlap=c_over,
                embeddings=embeddings
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

        # Evaluate across all 50 questions
        hit1_count = 0
        hit3_count = 0
        hit5_count = 0
        mrr_sum = 0.0
        prec_sum = 0.0
        kw_cov_sum = 0.0
        latencies = []

        print(f"\nEvaluating {cfg_name} (k={top_k}, type={ret_type}, lenient={is_lenient})...")
        for test in test_cases:
            q = test["question"]
            kws = test["gold_keywords"]

            t0 = time.perf_counter()
            docs = retriever.invoke(q)
            lat_ms = (time.perf_counter() - t0) * 1000
            latencies.append(lat_ms)

            met = compute_retrieval_metrics(docs, kws, k_values=(1, 3, 5), lenient=is_lenient)
            hit1_count += met["hit@1"]
            hit3_count += met["hit@3"]
            hit5_count += met["hit@5"]
            mrr_sum += met["mrr"]
            prec_sum += met["precision"]
            kw_cov_sum += met["keyword_coverage"]

        n = len(test_cases)
        avg_lat = sum(latencies) / n
        hit1_pct = round((hit1_count / n) * 100, 1)
        hit3_pct = round((hit3_count / n) * 100, 1)
        hit5_pct = round((hit5_count / n) * 100, 1)
        mrr = round(mrr_sum / n, 3)
        kw_cov = round((kw_cov_sum / n) * 100, 1)

        result_row = {
            "Configuration": cfg_name,
            "Chunk Size": c_size,
            "Overlap": c_over,
            "Top K": top_k,
            "Hit@1 (%)": hit1_pct,
            "Hit@3 (%)": hit3_pct,
            "Hit@5 (%)": hit5_pct,
            "MRR": mrr,
            "Keyword Recall (%)": kw_cov,
            "Latency (ms)": round(avg_lat, 1)
        }
        summary_rows.append(result_row)
        print(f"  → Hit@1: {hit1_pct}% | Hit@5: {hit5_pct}% | MRR: {mrr} | Latency: {avg_lat:.1f}ms")

    # Output Markdown and CSV
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_file = out_path / f"ablation_matrix_{ts}.md"
    csv_file = out_path / f"ablation_matrix_{ts}.csv"

    # Write Markdown Table
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Janani AI - RAG Parameter Ablation & Optimization Matrix\n\n")
        f.write(f"**Evaluated on 50 Gold-Standard Clinical Questions (FOGSI Medical Guidelines)**\n\n")
        f.write("| Configuration | Chunk Size | Overlap | Top K | Hit@1 | Hit@3 | Hit@5 | MRR | Keyword Recall | Latency (ms) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in summary_rows:
            f.write(f"| **{r['Configuration']}** | {r['Chunk Size']} | {r['Overlap']} | {r['Top K']} | {r['Hit@1 (%)']}% | {r['Hit@3 (%)']}% | {r['Hit@5 (%)']}% | **{r['MRR']}** | {r['Keyword Recall (%)']}% | {r['Latency (ms)']} ms |\n")

    # Write CSV
    import csv
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "="*70)
    print("🏆 FINAL ABLATION MATRIX")
    print("="*70)
    print(f"{'Configuration':<26} | {'Hit@1':<7} | {'Hit@5':<7} | {'MRR':<6} | {'Keyword':<8} | {'Latency':<8}")
    print("-" * 70)
    for r in summary_rows:
        print(f"{r['Configuration']:<26} | {str(r['Hit@1 (%)']) + '%':<7} | {str(r['Hit@5 (%)']) + '%':<7} | {r['MRR']:<6} | {str(r['Keyword Recall (%)']) + '%':<8} | {str(r['Latency (ms)']) + 'ms':<8}")
    print("="*70 + "\n")
    print(f"Results saved to:\n  - {md_file}\n  - {csv_file}")

    return summary_rows


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    testset_file = str(Path(__file__).parent / "testset.json")
    health_file = str(base_dir / "health_book.txt")
    out_dir = str(Path(__file__).parent / "results")

    run_ablation_experiment(
        testset_path=testset_file,
        health_book_path=health_file,
        output_dir=out_dir
    )
