---
title: RAG Research Paper Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAG Research Paper Assistant

Upload research papers (or fetch them from arXiv) and chat with them. Answers are grounded in the
papers and cite paper name + page.

**Pipeline:** PyMuPDF → chunking → BGE embeddings → ChromaDB (HNSW, cosine) + BM25 → Reciprocal Rank Fusion
→ cross-encoder reranker → Llama 3 (Groq) → cited answer.

**Features:** multi-paper chat, follow-up question rewriting, paper summary, paper comparison, arXiv search,
evaluation (Recall@k, MRR) with ablation study.

## Run locally
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# create .env with GROQ_API_KEY=...
streamlit run app.py
```
