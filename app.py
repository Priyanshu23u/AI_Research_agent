import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main Page Search UI
st.title("🔬 AI Research Assistant")
st.markdown("Use **Local embeddings** + DuckDuckGo Search to find and analyze research papers.")

search_query = st.text_input("📄 Search Topic / Keywords", "quantum computing")
max_results = st.slider("Number of papers", 5, 50, 10)
sort_order = st.selectbox("Sort by", ["Relevance", "Last Updated", "Citation Count"])
export_format = st.selectbox("Export Format", ["PDF", "Markdown"])

if st.button("🔍 Search Papers"):
    st.session_state["trigger_search"] = True

# Load local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    try:
        emb = embedding_model.encode(text, convert_to_numpy=True)
        return emb
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def fetch_papers_duckduckgo(query, max_results=10):
    results = DDGS().text(query, max_results=max_results)
    papers = []
    if not results:
        return papers
    for r in results:
        title = r.get("title", "")
        snippet = r.get("body", "")
        url = r.get("href", "")
        arxiv_id = None
        if "arxiv.org/abs/" in url:
            arxiv_id = url.split("arxiv.org/abs/")[-1].split("#")[0].split("?")[0]
        papers.append({
            "title": title,
            "summary": snippet,
            "link": url,
            "arxiv_id": arxiv_id
        })
    return papers

def fetch_arxiv_metadata(arxiv_id):
    if not arxiv_id:
        return None
    try:
        base_url = "http://export.arxiv.org/api/query"
        params = {"id_list": arxiv_id}
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        published = entry.find("atom:published", ns).text.strip()
        authors = [a.text for a in entry.findall("atom:author/atom:name", ns)]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return {
            "title": title,
            "summary": summary,
            "authors": ", ".join(authors) if authors else None,
            "published": published if published else None,
            "pdf_url": pdf_url,
            "link": f"https://arxiv.org/abs/{arxiv_id}"
        }
    except Exception:
        return None

# --- Initialize papers in session state ---
if "papers" not in st.session_state:
    st.session_state["papers"] = []

# Main paper fetching
if "trigger_search" in st.session_state and st.session_state["trigger_search"]:
    st.session_state["trigger_search"] = False
    with st.spinner("🔍 Searching papers via DuckDuckGo..."):
        raw_papers = fetch_papers_duckduckgo(search_query, max_results)
        if not raw_papers:
            st.warning("No papers found. Try different keywords.")
        else:
            st.success(f"✅ Found {len(raw_papers)} papers.")

            enriched_papers = []
            for p in raw_papers:
                if p.get("arxiv_id"):
                    meta = fetch_arxiv_metadata(p["arxiv_id"])
                    if meta:
                        enriched_papers.append(meta)
                        continue
                enriched_papers.append({
                    "title": p["title"],
                    "summary": p["summary"],
                    "authors": None,
                    "published": None,
                    "pdf_url": p["link"],
                    "link": p["link"],
                })

            for p in enriched_papers:
                emb = embed_text(p["summary"])
                p["embedding"] = emb

            # Store in session state
            st.session_state["papers"] = enriched_papers

            # ✅ Display results cleanly
            st.subheader("📑 Search Results")
            for r in st.session_state["papers"]:
                st.markdown(f"### {r['title']}")
                if r.get("authors"):
                    st.write(f"**Authors:** {r['authors']}")
                if r.get("published"):
                    st.write(f"**Published:** {r['published']}")
                st.write(f"**Summary:** {r['summary']}")
                st.markdown(f"[📄 Read PDF]({r['pdf_url']})", unsafe_allow_html=True)
                st.markdown("---")

def semantic_search(query, papers, top_k=5):
    query_emb = embed_text(query)
    if query_emb is None:
        return []
    scored = []
    for paper in papers:
        if paper.get("embedding") is not None:
            score = np.dot(query_emb, paper["embedding"]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(paper["embedding"])
            )
            scored.append((score, paper))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in scored[:top_k]]

# Semantic Search UI
st.subheader("🔍 Semantic Search in Papers")
semantic_query = st.text_input("Enter your research query for semantic search:")

if st.button("Search Semantic") and semantic_query:
    if not st.session_state["papers"]:
        st.warning("Please search papers first using the main search.")
    else:
        results = semantic_search(semantic_query, st.session_state["papers"])
        if results:
            st.success(f"Found {len(results)} relevant papers:")
            for r in results:
                st.markdown(f"### {r['title']}")
                if r.get("authors"):
                    st.write(f"**Authors:** {r['authors']}")
                if r.get("published"):
                    st.write(f"**Published:** {r['published']}")
                st.write(f"**Summary:** {r['summary']}")
                st.markdown(f"[📄 Read PDF]({r['pdf_url']})", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning("No relevant papers found.")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.write("""
This AI Research Assistant uses:
- **DuckDuckGo** for paper discovery
- **arXiv API** for metadata enrichment
- **Local sentence-transformers** for semantic embeddings and search
- **Streamlit** for UI
""")
st.sidebar.write("🔬 Completely free, no paid API needed.")
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")

# Custom CSS
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #161a23;
    }
    .stTextInput input, .stButton button {
        background-color: #262730;
        color: #ffffff;
        border-radius: 6px;
    }
    h1, h2, h3 {
        color: #f5f5f5;
    }
    a {
        color: #4db8ff !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)
