import os
import json
import argparse
import pandas as pd
from .arxiv_client import search_arxiv
from .library import Library
from .generator import LocalGenerator, DEFAULT_MODEL
from .paper_builder import make_prompt, refs_markdown, assemble_markdown, write_docx, write_pdf
from .enhanced_analyzer import EnhancedPaperAnalyzer
from .enhanced_paper_builder import EnhancedPaperBuilder

OUT_DIR = os.path.join("outputs")
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


def cmd_search(args):
    print(f"Searching arXiv for: {args.query}")
    papers = search_arxiv(args.query, max_results=args.max)
    print(f"Found {len(papers)} candidates.")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"Saved candidates → {args.out}")


def cmd_add(args):
    lib = Library(embed_model=args.embed_model)
    with open(args.from_json, "r", encoding="utf-8") as f:
        cands = json.load(f)

    if args.select:
        idxs = [int(x) for x in args.select.split(",") if x.strip()]
        chosen = [cands[i] for i in idxs if 0 <= i < len(cands)]
    else:
        chosen = cands

    added = lib.add_papers(chosen)
    print(f"Added {added} papers to library.")


def cmd_list(args):
    lib = Library(embed_model=args.embed_model)
    df = lib.list()
    if df.empty:
        print("Library is empty.")
        return
    for i, row in df.reset_index().iterrows():
        print(f"[{i}] {row.get('title','')} — {row.get('year','')} — {row.get('authors','')}")


def cmd_generate(args):
    lib = Library(embed_model=args.embed_model)

    if args.use_ids:
        ids = [int(x) for x in args.use_ids.split(",") if x.strip()]
        papers_df = lib.get_by_ids(ids)
        if papers_df.empty:
            print("Selected IDs not found in library.")
            return
    else:
        if not args.topic:
            raise SystemExit("--topic is required for retrieval mode.")
        papers_df = lib.retrieve(args.topic, top_k=args.top_k)

    print(f"Using {len(papers_df)} papers to generate...")
    gen = LocalGenerator(model_name=args.gen_model, load_in_4bit=args.load_in_4bit)

    sections = {}
    for sec, req in SECTION_REQS.items():
        prompt = make_prompt(args.topic, papers_df, sec, req)
        text = gen.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        sections[sec] = text
        print(f"Generated: {sec} ({len(text)} chars)")

    refs_md = refs_markdown(papers_df)
    title = f"{args.topic} — A Survey" if args.topic else "Research Survey"
    md = assemble_markdown(sections, refs_md, title)

    md_path = os.path.join(OUT_DIR, f"{args.out_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved Markdown → {md_path}")

    if args.docx:
        write_docx(os.path.join(OUT_DIR, f"{args.out_name}.docx"), title, sections, refs_md)
        print(f"Saved DOCX → ./outputs/{args.out_name}.docx")

    if args.pdf:
        write_pdf(os.path.join(OUT_DIR, f"{args.out_name}.pdf"), title, sections, refs_md)
        print(f"Saved PDF → ./outputs/{args.out_name}.pdf")


def build_parser():
    p = argparse.ArgumentParser(description="AI Research Paper Builder (Free, Local)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Search
    ps = sub.add_parser("search", help="Search arXiv and save candidates JSON")
    ps.add_argument("--query", required=True)
    ps.add_argument("--max", type=int, default=50)
    ps.add_argument("--out", default="candidates.json")
    ps.set_defaults(func=cmd_search)

    # Add
    pa = sub.add_parser("add", help="Add selected candidates to vector library")
    pa.add_argument("--from-json", required=True)
    pa.add_argument("--select", default="", help="Comma-separated 0-based indexes (empty = all)")
    pa.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    pa.set_defaults(func=cmd_add)

    # List
    pl = sub.add_parser("list", help="List stored library items with IDs")
    pl.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    pl.set_defaults(func=cmd_list)

    # Generate
    pg = sub.add_parser("generate", help="Generate research paper")
    pg.add_argument("--topic", default="")
    pg.add_argument("--use-ids", default="")
    pg.add_argument("--top-k", type=int, default=12)
    pg.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    pg.add_argument("--gen-model", default=DEFAULT_MODEL)
    pg.add_argument("--load-in-4bit", action="store_true")
    pg.add_argument("--out-name", default="research_survey")
    pg.add_argument("--max-new-tokens", type=int, default=1100)
    pg.add_argument("--temperature", type=float, default=0.5)
    pg.add_argument("--top-p", type=float, default=0.9)
    pg.add_argument("--docx", action="store_true")
    pg.add_argument("--pdf", action="store_true")
    pg.set_defaults(func=cmd_generate)

    return p
def cmd_generate_enhanced(args):
    """Enhanced generation with comprehensive analysis"""
    lib = Library(embed_model=args.embed_model)
    
    if args.use_ids:
        ids = [int(x) for x in args.use_ids.split(",") if x.strip()]
        papers_df = lib.get_by_ids(ids)
    else:
        papers_df = lib.retrieve(args.topic, top_k=args.top_k)
    
    if papers_df.empty:
        print("No papers found.")
        return
    
    # Convert DataFrame to list of dicts
    papers = papers_df.to_dict('records')
    
    # Initialize enhanced components
    gen = LocalGenerator(model_name=args.gen_model, load_in_4bit=args.load_in_4bit)
    analyzer = EnhancedPaperAnalyzer(gen)
    builder = EnhancedPaperBuilder(gen)
    
    # Analyze papers
    print("Analyzing papers...")
    analyses = analyzer.analyze_papers_batch(papers)
    
    # Identify gaps
    print("Identifying research gaps...")
    gaps = analyzer.identify_research_gaps(analyses)
    
    # Generate trends
    print("Analyzing trends...")
    trends = analyzer.generate_research_trends(analyses)
    
    # Generate enhanced paper
    print("Generating enhanced research paper...")
    sections = builder.generate_enhanced_paper(args.topic, analyses, gaps, trends)
    
    # Save results
    refs_md = refs_markdown(papers_df)
    title = f"{args.topic} — A Comprehensive Survey"
    md = assemble_markdown(sections, refs_md, title)
    
    md_path = os.path.join(OUT_DIR, f"{args.out_name}_enhanced.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    print(f"✅ Enhanced paper saved to {md_path}")

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
