"""
Auto-creates an evaluation set: picks random chunks and asks the LLM
to write a question that this chunk answers. Run AFTER indexing papers.
Usage:  python generate_eval.py
"""
import json
import os
import random
import time
import config
from rag import RAG

N_QUESTIONS = 30

rag = RAG()
random.seed(42)

candidates = [i for i, d in enumerate(rag.docs) if len(d) > 600]
sample = random.sample(candidates, min(N_QUESTIONS, len(candidates)))

out = []
for n, i in enumerate(sample, start=1):
    prompt = f"""Read this excerpt from a research paper and write ONE specific question
that can be answered using ONLY this excerpt.
Rules: the question must be self-contained (never say "the passage" or "the excerpt"),
and should paraphrase instead of copying exact sentences.
Return only the question.

Excerpt:
{rag.docs[i]}"""
    try:
        q = rag._llm(prompt, temperature=0.3).strip()
    except Exception as e:
        print("skip:", e)
        time.sleep(5)
        continue
    out.append({
        "question": q,
        "chunk_id": rag.ids[i],
        "paper": rag.metas[i]["paper"],
        "page": rag.metas[i]["page"],
    })
    print(f"[{n}/{len(sample)}] {q}")
    time.sleep(2)  # stay inside free-tier rate limits

os.makedirs("data", exist_ok=True)
with open(config.EVAL_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(out)} questions to {config.EVAL_FILE}")
