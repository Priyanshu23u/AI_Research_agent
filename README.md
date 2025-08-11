# 🔬 AI Research Assistant

A **Streamlit-based research assistant** that helps you:
- Search for academic papers using **DuckDuckGo** and **arXiv API**
- Store and reuse search results for **semantic search** with local embeddings
- Analyze and filter papers based on meaning, not just keywords
- Export results in multiple formats (planned: PDF, Markdown)

---

## ✨ Features
- **DuckDuckGo Search** → Finds relevant research papers without requiring API keys
- **arXiv Metadata Enrichment** → Fetches authors, abstracts, publication dates, and PDF links when available
- **Local Embedding Model** → Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to create paper summaries and semantic search vectors
- **Semantic Search** → Finds papers most relevant to your research question using cosine similarity
- **Streamlit UI** → Interactive search, filtering, and results display
- **Dark Theme Styling** → Custom CSS for better visual experience

---

## 🛠 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant


### Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows

pip install -r requirements.txt

### Usage
streamlit run app.py
