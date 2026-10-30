import os
import streamlit as st
import config
from rag import RAG
from arxiv_search import search_arxiv, download_pdf

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Research Paper Assistant")

if not config.GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file (or Space secrets) and restart.")
    st.stop()


@st.cache_resource(show_spinner="Loading models (first time takes a few minutes)...")
def load_rag():
    return RAG()


rag = load_rag()
os.makedirs(config.PAPERS_DIR, exist_ok=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📄 Your papers")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index papers", use_container_width=True) and files:
        for f in files:
            path = os.path.join(config.PAPERS_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            with st.spinner(f"Indexing {f.name}..."):
                n = rag.add_pdf(path)
            if n:
                st.success(f"{f.name}: {n} chunks")
            else:
                st.warning(f"{f.name}: no text found (scanned PDF?)")
        st.rerun()

    papers = rag.list_papers()
    st.caption(f"{len(papers)} paper(s) indexed")
    selected = st.multiselect("Search in", papers, default=papers)

    if papers:
        with st.expander("🗑️ Remove a paper"):
            to_del = st.selectbox("Paper", papers)
            if st.button("Delete"):
                rag.delete_paper(to_del)
                st.rerun()

tab_chat, tab_sum, tab_cmp, tab_arxiv = st.tabs(
    ["💬 Chat", "📝 Summary", "⚖️ Compare", "🔎 Find on arXiv"]
)


def show_sources(sources):
    with st.expander("Sources used"):
        for s in sources:
            score = f" (score {s['score']:.2f})" if s.get("score") is not None else ""
            st.markdown(f"**{s['paper']}, page {s['page']}**{score}")
            st.caption(s["text"][:300] + "...")


# ---------------- Chat ----------------
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (role, text, sources)

    for role, text, sources in st.session_state.history:
        with st.chat_message(role):
            st.write(text)
            if sources:
                show_sources(sources)

    question = st.chat_input("Ask something about your papers...")
    if question:
        st.chat_message("user").write(question)
        plain_hist = [(r, t) for r, t, _ in st.session_state.history]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources, standalone = rag.answer(question, selected, plain_hist)
                except Exception as e:
                    answer, sources, standalone = f"Error: {e}", [], question
            st.write(answer)
            if standalone != question:
                st.caption(f"🔁 Searched as: *{standalone}*")
            if sources:
                show_sources(sources)
        st.session_state.history += [("user", question, None), ("assistant", answer, sources)]

    if st.session_state.history and st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# ---------------- Summary ----------------
with tab_sum:
    if not papers:
        st.info("Upload a paper first.")
    else:
        p = st.selectbox("Choose a paper", papers, key="sum_paper")
        if st.button("Generate summary"):
            with st.spinner("Reading the paper..."):
                try:
                    summary, sources = rag.summarize(p)
                    st.markdown(summary)
                    show_sources(sources)
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------------- Compare ----------------
with tab_cmp:
    if len(papers) < 2:
        st.info("Index at least 2 papers to compare.")
    else:
        chosen = st.multiselect("Papers to compare (2-3)", papers, max_selections=3, key="cmp_papers")
        aspect = st.text_input("Compare on", "proposed method and results")
        if st.button("Compare") and len(chosen) >= 2:
            with st.spinner("Comparing..."):
                try:
                    result, sources = rag.compare(chosen, aspect)
                    st.markdown(result)
                    show_sources(sources)
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------------- arXiv ----------------
with tab_arxiv:
    q = st.text_input("Search arXiv", placeholder="e.g. retrieval augmented generation survey")
    if st.button("Search") and q:
        with st.spinner("Searching arXiv..."):
            try:
                st.session_state.arxiv = search_arxiv(q)
            except Exception as e:
                st.error(f"arXiv search failed: {e}")

    for item in st.session_state.get("arxiv", []):
        with st.expander(f"{item['title']} ({item['year']})"):
            st.caption(item["authors"])
            st.write(item["summary"])
            if st.button("➕ Add to my library", key=f"add_{item['id']}"):
                with st.spinner("Downloading and indexing..."):
                    try:
                        path = download_pdf(item, config.PAPERS_DIR)
                        n = rag.add_pdf(path)
                        st.success(f"Added ({n} chunks). Open the sidebar to select it.")
                    except Exception as e:
                        st.error(f"Failed: {e}")
