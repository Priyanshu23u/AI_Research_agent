import os
import streamlit as st
import pandas as pd

from src.arxiv_client import search_arxiv
from src.library import Library
from src.generator import LocalGenerator, DEFAULT_MODEL
from src.paper_builder import make_prompt, refs_markdown, assemble_markdown, write_docx, write_pdf

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="AI Research Paper Builder", layout="wide")
st.title("📄 AI Research Paper Builder (Free, Local)")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SECTION_REQS = {
    "Abstract": "200–300 words, concise overview of topic, emphasize contributions and trends.",
    "Introduction": "Define the problem, scope, motivation, and outline of the survey.",
    "Literature Review": "Summarize prior work, compare methods, datasets, metrics; highlight gaps.",
    "Key Findings": "Bullet or paragraph form; group by themes; include notable methods and results.",
    "Results & Discussion": "Synthesize outcomes, trade-offs, limitations; discuss significance and challenges.",
    "Conclusion": "Summarize insights; restate contributions; provide final takeaways.",
    "Future Works": "Propose concrete open problems, datasets, evaluation improvements, and research directions.",
}

# ---------------------------
# Session State Initialization
# ---------------------------
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_search" not in st.session_state:
    st.session_state.selected_search = set()
if "library" not in st.session_state:
    st.session_state.library = Library()
if "generator" not in st.session_state:
    st.session_state.generator = None

lib = st.session_state.library

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📚 Library", "📝 Generate", "⚙ Settings"])

# ---------------------------
# Search Tab
# ---------------------------
with tab1:
    st.subheader("🔍 Search arXiv")
    qcol, mcol = st.columns(2)
    with qcol:
        query = st.text_input("Enter your research query:", placeholder="e.g., AI fairness in healthcare")
    with mcol:
        max_results = st.number_input("Max results", 5, 100, 25, step=5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                with st.spinner("Searching arXiv..."):
                    papers = search_arxiv(query, max_results=max_results)
                st.session_state.search_results = [{
                    "title": p.get("title", ""),
                    "authors": [a.strip() for a in p.get("authors", "").split(",") if a.strip()],
                    "summary": p.get("abstract", ""),
                    "published": p.get("published", ""),
                    "year": p.get("year", ""),
                    "link": p.get("link_abs") or p.get("link_pdf") or "",
                    "link_pdf": p.get("link_pdf", ""),
                    "id": p.get("id", ""),
                    "categories": p.get("categories", ""),
                } for p in papers]
                st.session_state.selected_search = set()
            except Exception as e:
                st.error(f"Search failed: {e}")

    if st.session_state.search_results:
        st.caption("Expand items and tick boxes to add to your library.")
        for idx, paper in enumerate(st.session_state.search_results):
            with st.expander(f"{idx}. {paper['title']}"):
                st.write(f"Authors: {', '.join(paper['authors'])}")
                st.write(f"Published: {paper.get('published','')}")
                st.write(f"Summary: {paper['summary']}")
                if paper["link"]:
                    st.write(f"[arXiv Link]({paper['link']})")
                checked = st.checkbox("Select this paper", key=f"sel_{idx}")
                if checked:
                    st.session_state.selected_search.add(idx)
                else:
                    st.session_state.selected_search.discard(idx)

        if st.button("➕ Add selected to library"):
            chosen = [st.session_state.search_results[i] for i in sorted(list(st.session_state.selected_search))]
            to_add = [{
                "id": c.get("id", ""),
                "title": c.get("title", ""),
                "abstract": c.get("summary", ""),
                "authors": c.get("authors", []),
                "published": c.get("published", ""),
                "year": c.get("year", ""),
                "link_abs": c.get("link", ""),
                "link_pdf": c.get("link_pdf", ""),
                "categories": c.get("categories", ""),
            } for c in chosen]
            try:
                added = lib.add_papers(to_add)
                st.success(f"Added {added} papers to the library.")
            except Exception as e:
                st.error(f"Failed to add to library: {e}")

# ---------------------------
# Library Tab
# ---------------------------
with tab2:
    st.subheader("📚 Your Library")
    try:
        df = lib.list()
        if df.empty:
            st.info("Library is empty. Use the Search tab to add papers.")
        else:
            show_cols = ["title", "authors", "year", "link_abs"]
            present_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(
                df[present_cols].reset_index().rename(columns={"index": "ID"}),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Failed to list library: {e}")

# ---------------------------
# Generate Tab
# ---------------------------
with tab3:
    st.subheader("📝 Generate Research Paper")
    gcol0, gcol1, gcol2, gcol3 = st.columns([2, 2, 1.5, 1.5])
    with gcol0:
        gen_model = st.text_input("Local LLM model", DEFAULT_MODEL, help="HF model id, e.g., microsoft/Phi-3-mini-4k-instruct")
    with gcol1:
        use_4bit = st.checkbox("Load in 4-bit (saves VRAM)", value=True, help="Requires bitsandbytes.")
    with gcol2:
        max_new_tokens = st.slider("Max new tokens/section", 300, 1600, 900, step=100)
    with gcol3:
        temperature = st.slider("Temperature", 0.0, 1.2, 0.5, step=0.05)

    topic = st.text_input("Paper Topic:", placeholder="Exact topic/title for your paper")
    mode = st.radio("Source of papers", ["Use specific stored IDs", "Retrieve top-k from library"], horizontal=True)

    ids_input = ""
    top_k = 12
    if mode == "Use specific stored IDs":
        ids_input = st.text_input("Enter comma-separated library IDs (as shown in the Library table):", "")
    else:
        top_k = st.slider("Top-k papers to retrieve by similarity", 3, 30, 12)

    out_fmt = st.radio("Output format", ["Markdown", "DOCX", "PDF"], horizontal=True)
    out_name = st.text_input("Output filename (no extension)", "research_survey")

    def get_generator():
        if st.session_state.generator is None:
            with st.spinner("Loading local LLM..."):
                try:
                    st.session_state.generator = LocalGenerator(model_name=gen_model, load_in_4bit=use_4bit)
                except Exception as e:
                    st.session_state.generator = None
                    st.error(f"Failed to load model: {e}")
        return st.session_state.generator

    if st.button("🚀 Generate"):
        if not topic.strip():
            st.warning("Please provide a topic.")
        else:
            # Select papers
            if mode == "Use specific stored IDs":
                try:
                    ids = [int(x.strip()) for x in ids_input.split(",") if x.strip()]
                except ValueError:
                    ids = []
                    st.error("IDs must be integers separated by commas.")
                df_sel = lib.get_by_ids(ids) if ids else pd.DataFrame([])
                if df_sel.empty:
                    st.error("No valid IDs selected or not found in library.")
                    st.stop()
                papers_df = df_sel
            else:
                try:
                    papers_df = lib.retrieve(topic, top_k=top_k)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")
                    st.stop()

            # Load generator
            gen = get_generator()
            if gen is None:
                st.stop()

            # Generate sections
            sections = {}
            for sec, req in SECTION_REQS.items():
                prompt = make_prompt(topic, papers_df, sec, req)
                with st.spinner(f"Generating {sec}..."):
                    try:
                        text = gen.generate(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=0.9
                        )
                    except Exception as e:
                        st.error(f"Generation failed for section '{sec}': {e}")
                        st.stop()
                sections[sec] = text

            refs_md = refs_markdown(papers_df)
            title = f"{topic} — A Survey"
            md = assemble_markdown(sections, refs_md, title)

            # Save markdown
            try:
                os.makedirs(OUT_DIR, exist_ok=True)
                md_path = os.path.join(OUT_DIR, f"{out_name}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md)
                st.success(f"✅ Generated Markdown: {md_path}")
                with open(md_path, "rb") as f:
                    st.download_button("⬇ Download .md", f, file_name=f"{out_name}.md", mime="text/markdown")
            except Exception as e:
                st.error(f"Failed to save Markdown: {e}")

            # Save DOCX
            if out_fmt == "DOCX":
                try:
                    docx_path = os.path.join(OUT_DIR, f"{out_name}.docx")
                    write_docx(docx_path, title, sections, refs_md)
                    st.success(f"✅ Generated DOCX: {docx_path}")
                    with open(docx_path, "rb") as f:
                        st.download_button("⬇ Download .docx", f, file_name=f"{out_name}.docx")
                except Exception as e:
                    st.error(f"Failed to generate DOCX: {e}")

            # Save PDF
            if out_fmt == "PDF":
                try:
                    pdf_path = os.path.join(OUT_DIR, f"{out_name}.pdf")
                    write_pdf(pdf_path, title, sections, refs_md)
                    st.success(f"✅ Generated PDF: {pdf_path}")
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇ Download .pdf", f, file_name=f"{out_name}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

# ---------------------------
# Settings Tab
# ---------------------------
with tab4:
    st.subheader("⚙ Settings")
    st.caption("All data stays local on your machine.")

    try:
        import torch
        gpu_info = "CUDA available: " + str(torch.cuda.is_available())
        if torch.cuda.is_available():
            gpu_info += f" | Device count: {torch.cuda.device_count()} | GPU: {torch.cuda.get_device_name(0)}"
        st.text(gpu_info)
    except Exception:
        st.text("PyTorch not installed or failed to query CUDA.")

    if st.button("🧹 Clear Library (danger)"):
        try:
            lib.clear()
            st.warning("Library cleared. FAISS index and metadata reset.")
        except Exception as e:
            st.error(f"Failed to clear library: {e}")

    if st.button("♻ Reload Model"):
        st.session_state.generator = None
        st.success("Model will reload on next generation request.")
