import os
import re
import arxiv
import requests


def search_arxiv(query, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for r in client.results(search):
        results.append({
            "id": r.get_short_id(),
            "title": r.title.strip().replace("\n", " "),
            "authors": ", ".join(a.name for a in r.authors[:4]),
            "year": r.published.year,
            "summary": r.summary.strip().replace("\n", " "),
            "pdf_url": r.pdf_url,
        })
    return results


def download_pdf(item, folder):
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", item["title"])[:60].strip("_")
    path = os.path.join(folder, f"{safe}.pdf")
    resp = requests.get(item["pdf_url"], timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path
