# AI Research Paper Builder (Free, Local)


This project lets you:
1) Search arXiv for papers (free API)
2) Store chosen papers into a local vector database (FAISS + Parquet)
3) Generate a full research paper (Abstract, Introduction, Literature Review, Key Findings, Results & Discussion, Conclusion, Future Works, References)
using a **local, free LLM** (default: microsoft/Phi-3-mini-4k-instruct).


### Install
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
#.\.venv\Scripts\activate
pip install -r requirements.txt
```


### CLI examples
```bash
# 1) Search and save candidates to a JSON file
python -m src.cli search --query "AI fairness in healthcare" --max 40 --out candidates.json


# 2) Add selected results (by their 0-based indexes in the JSON) to your library
python -m src.cli add --from-json candidates.json --select 0,2,5


# 3) List your library contents (IDs are stable and used later)
python -m src.cli list


# 4) Generate from specific stored IDs
python -m src.cli generate --topic "AI fairness in healthcare" --use-ids 3,7,11 --out-name fairness_survey


# 5) Or generate by retrieving top‑k similar from your library
python -m src.cli generate --topic "AI fairness in healthcare" --top-k 12 --out-name fairness_survey
```


All outputs are saved to `./outputs/`. Library data (FAISS + metadata parquet) is kept in `./data/library/`.