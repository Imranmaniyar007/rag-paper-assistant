import os
import re
import fitz  # PyMuPDF
import chromadb
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Cut text into overlapping pieces."""
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if len(piece) > 50:
            chunks.append(piece)
        start += size - overlap
    return chunks


class RAG:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)
        self.llm = Groq(api_key=config.GROQ_API_KEY)

        os.makedirs(config.DB_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=config.DB_DIR)
        self.col = client.get_or_create_collection(
            "papers", metadata={"hnsw:space": "cosine"}
        )
        self._build_bm25()

    # ---------- LLM helper ----------
    def _llm(self, prompt, temperature=0.2):
        resp = self.llm.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content

    # ---------- INDEXING ----------
    def add_pdf(self, pdf_path):
        paper = os.path.basename(pdf_path)
        self.col.delete(where={"paper": paper})  # avoid duplicates on re-upload

        doc = fitz.open(pdf_path)
        docs, metas, ids = [], [], []
        for page_no, page in enumerate(doc, start=1):
            for i, chunk in enumerate(chunk_text(page.get_text())):
                docs.append(chunk)
                metas.append({"paper": paper, "page": page_no})
                ids.append(f"{paper}-p{page_no}-c{i}")

        if not docs:  # e.g. scanned PDF with no text layer
            self._build_bm25()
            return 0
        embs = self.embedder.encode(docs, normalize_embeddings=True).tolist()
        self.col.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
        self._build_bm25()
        return len(docs)

    def delete_paper(self, paper):
        self.col.delete(where={"paper": paper})
        path = os.path.join(config.PAPERS_DIR, paper)
        if os.path.exists(path):
            os.remove(path)
        self._build_bm25()

    def _build_bm25(self):
        data = self.col.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        self.id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        tokenized = [tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def list_papers(self):
        return sorted({m["paper"] for m in self.metas})

    # ---------- RETRIEVAL ----------
    def _pack(self, idx_list, scores=None):
        out = []
        for n, i in enumerate(idx_list):
            out.append({
                "id": self.ids[i],
                "text": self.docs[i],
                "paper": self.metas[i]["paper"],
                "page": self.metas[i]["page"],
                "score": float(scores[n]) if scores is not None else None,
            })
        return out

    def retrieve(self, question, papers=None, mode="hybrid_rerank", top_k=None):
        """
        mode = "dense"          -> embeddings only
               "hybrid"         -> embeddings + BM25 merged with RRF
               "hybrid_rerank"  -> hybrid + cross-encoder reranker (best)
        """
        top_k = top_k or config.TOP_K_FINAL
        if not self.docs:
            return []
        allowed = [i for i, m in enumerate(self.metas)
                   if not papers or m["paper"] in papers]
        if not allowed:
            return []       
        n = config.TOP_K_RETRIEVE

        # 1) Dense (meaning) search
        q_emb = self.embedder.encode(
            QUERY_PREFIX + question, normalize_embeddings=True
        ).tolist()
        where = {"paper": {"$in": list(papers)}} if papers else None
        dense = self.col.query(
            query_embeddings=[q_emb],
            n_results=min(n, len(allowed)),
            where=where,
        )
        dense_idx = [self.id_to_idx[i] for i in dense["ids"][0]]
        if mode == "dense":
            return self._pack(dense_idx[:top_k])

        # 2) BM25 (keyword) search
        scores = self.bm25.get_scores(tokenize(question))
        bm25_idx = sorted(allowed, key=lambda i: scores[i], reverse=True)[:n]

        # 3) Reciprocal Rank Fusion
        fused = {}
        for ranking in (dense_idx, bm25_idx):
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0) + 1 / (config.RRF_K + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:n]
        if mode == "hybrid":
            return self._pack(candidates[:top_k])

        # 4) Rerank with cross-encoder
        pairs = [(question, self.docs[i]) for i in candidates]
        rr = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, rr), key=lambda x: x[1], reverse=True)[:top_k]
        return self._pack([i for i, _ in ranked], [s for _, s in ranked])

    # ---------- GENERATION ----------
    def rewrite(self, question, history):
        """Turn a follow-up like 'what about its dataset?' into a standalone question."""
        if not history:
            return question
        recent = history[-config.HISTORY_TURNS:]
        convo = "\n".join(f"{role}: {text}" for role, text in recent)
        prompt = f"""Given the conversation and a follow-up question, rewrite the follow-up
as one standalone question that makes sense without the conversation.
If it is already standalone, return it unchanged. Return ONLY the question.

Conversation:
{convo}

Follow-up: {question}
Standalone question:"""
        try:
            return self._llm(prompt, temperature=0).strip().strip('"')
        except Exception:
            return question

    def answer(self, question, papers=None, history=None):
        standalone = self.rewrite(question, history or [])
        chunks = self.retrieve(standalone, papers)
        if not chunks:
            return "Please upload a paper first.", [], standalone

        context = "\n\n".join(
            f"[{c['paper']}, p.{c['page']}]\n{c['text']}" for c in chunks
        )
        prompt = f"""You are a research paper assistant.
Answer ONLY using the context below. Cite sources like [paper.pdf, p.3].
If the answer is not in the context, say "I could not find this in the uploaded papers."

Context:
{context}

Question: {standalone}"""
        return self._llm(prompt), chunks, standalone

    def summarize(self, paper):
        """Structured summary built from the most relevant chunks of ONE paper."""
        topics = [
            "abstract and main contribution of the paper",
            "proposed method, model architecture and approach",
            "datasets, experiments and evaluation metrics",
            "results, comparison with baselines and conclusions",
            "limitations and future work",
        ]
        seen, chunks = set(), []
        for t in topics:
            for c in self.retrieve(t, [paper], top_k=3):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    chunks.append(c)
        if not chunks:
            return "No text found for this paper.", []

        context = "\n\n".join(f"[p.{c['page']}]\n{c['text']}" for c in chunks)
        prompt = f"""Summarize the research paper using ONLY the excerpts below.
Use these markdown headings: Problem, Method, Data & Experiments, Results, Limitations.
Keep each section to 2-4 sentences. If a section is not covered, write "Not found in excerpts".
Cite pages like (p.3).

Excerpts:
{context}"""
        return self._llm(prompt), chunks

    def compare(self, papers, aspect):
        """Compare several papers on one aspect (e.g. 'method', 'dataset', 'results')."""
        blocks, all_chunks = [], []
        for p in papers:
            cs = self.retrieve(aspect, [p], top_k=4)
            all_chunks += cs
            text = "\n".join(f"(p.{c['page']}) {c['text']}" for c in cs)
            blocks.append(f"=== PAPER: {p} ===\n{text}")

        prompt = f"""Compare the following research papers on this aspect: "{aspect}".
Use ONLY the excerpts. Output a markdown table (one row per paper) followed by a
3-4 sentence summary of the key similarities and differences.
Cite pages like (p.3). If something is missing, write "not stated".

{chr(10).join(blocks)}"""
        return self._llm(prompt), all_chunks
