"""
Ablation study: dense only vs hybrid vs hybrid + reranker.
Metrics: Recall@1, Recall@5, MRR@5.
Usage:  python eval.py
"""
import json
import pandas as pd
import config
from rag import RAG

rag = RAG()
with open(config.EVAL_FILE, encoding="utf-8") as f:
    data = json.load(f)

modes = {
    "Dense only": "dense",
    "Hybrid (Dense + BM25 + RRF)": "hybrid",
    "Hybrid + Reranker": "hybrid_rerank",
}

rows = []
for name, mode in modes.items():
    hit1 = hit5 = 0
    mrr = 0.0
    for item in data:
        res = rag.retrieve(item["question"], mode=mode, top_k=5)
        ids = [r["id"] for r in res]
        if item["chunk_id"] in ids:
            rank = ids.index(item["chunk_id"]) + 1
            hit5 += 1
            mrr += 1 / rank
            if rank == 1:
                hit1 += 1
    n = len(data)
    rows.append({
        "Method": name,
        "Recall@1": round(hit1 / n, 3),
        "Recall@5": round(hit5 / n, 3),
        "MRR@5": round(mrr / n, 3),
    })
    print("done:", name)

df = pd.DataFrame(rows)
print("\n", df.to_string(index=False))
df.to_csv("data/ablation_results.csv", index=False)
print("\nSaved to data/ablation_results.csv")
