# 🔍 LangChain Retrievers

> *Exploring different retrieval strategies in LangChain — from simple similarity search to advanced LLM-powered retrieval.*

---

## 📂 Project Structure

```
LangChainRetriever/
├── wikipedia_retriever.py        # 🌐 Wikipedia API-based retrieval
├── vectorStore_retriever.py      # 🗄️  Vector store similarity search (Chroma)
├── mmr_retriever.py              # ⚖️  Maximal Marginal Relevance (FAISS)
├── multiquery_retriever.py       # 🧠 LLM-powered multi-query retrieval (FAISS)
├── contexual_retriever.py        # 🎯 Contextual compression retrieval (Chroma)
└── README.md
```

---

## 📖 Retriever Breakdown

### 1️⃣ Wikipedia Retriever — `wikipedia_retriever.py`

> Fetches real-time information directly from **Wikipedia's API**.

| Detail | Value |
|--------|-------|
| **Source** | Wikipedia (live API) |
| **Package** | `langchain-community`, `wikipedia` |
| **Top K Results** | 2 |
| **Language** | English |

**How it works:** No embeddings or vector stores needed — it queries Wikipedia directly and returns the top matching articles as LangChain `Document` objects.

```python
retriever = WikipediaRetriever(top_k_results=2, lang='en')
docs = retriever.invoke("When did India win the first 50 overs World Cup?")
```

---

### 2️⃣ Vector Store Retriever — `vectorStore_retriever.py`

> Classic **similarity search** using Chroma as the vector store.

| Detail | Value |
|--------|-------|
| **Vector Store** | Chroma |
| **Embeddings** | Google Gemini (`gemini-embedding-2-preview`) |
| **Search Type** | Similarity |
| **Top K** | 2 |

**How it works:** Documents are embedded into vectors using Google's embedding model, stored in Chroma, and retrieved by finding the most similar vectors to the query.

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("What is chroma?")
```

---

### 3️⃣ MMR Retriever — `mmr_retriever.py`

> **Maximal Marginal Relevance** — balances relevance with diversity in results.

| Detail | Value |
|--------|-------|
| **Vector Store** | FAISS |
| **Embeddings** | Google Gemini (`gemini-embedding-2-preview`) |
| **Search Type** | MMR |
| **Top K** | 2 |
| **Lambda** | `0.5` (equal weight to relevance & diversity) |

**How it works:** Unlike pure similarity search, MMR actively avoids returning near-duplicate results. The `lambda_mult` parameter controls the trade-off:

- `lambda_mult = 1.0` → Pure similarity (no diversity)
- `lambda_mult = 0.0` → Maximum diversity
- `lambda_mult = 0.5` → Balanced (used here)

```python
retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 2, 'lambda_mult': 0.5}
)
```

> ⚠️ **Note:** On macOS Apple Silicon (M-series chips), FAISS requires the OpenMP fix:
> ```python
> import os
> os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
> ```

---

### 4️⃣ Multi-Query Retriever — `multiquery_retriever.py`

> Uses an **LLM to generate multiple query variations**, then combines results for better recall.

| Detail | Value |
|--------|-------|
| **Vector Store** | FAISS |
| **Embeddings** | Google Gemini (`gemini-embedding-2-preview`) |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **Search Type** | Similarity + LLM-enhanced |
| **Top K** | 5 |

**How it works:** A single user query like *"How to improve energy levels?"* gets transformed by the LLM into multiple variations (e.g., *"Ways to boost daily energy"*, *"Tips for maintaining physical balance"*). Each variation is searched independently, and results are merged — capturing documents that a single query might miss.

```python
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
    llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
)
```

The script compares **Standard Search** vs **MultiQueryRetriever** side-by-side to demonstrate the difference.

---

### 5️⃣ Contextual Compression Retriever — `contexual_retriever.py`

> Retrieves documents, then uses an **LLM to extract only the relevant parts** — compressing away noise.

| Detail | Value |
|--------|-------|
| **Vector Store** | Chroma |
| **Embeddings** | Google Gemini (`gemini-embedding-2-preview`) |
| **LLM (Compressor)** | Meta Llama 3.1 70B via HuggingFace |
| **Compressor** | `LLMChainExtractor` |
| **Top K** | 2 |

**How it works:** Documents often contain mixed content. For example, a document about the Grand Canyon might also mention photosynthesis. When you query *"What is photosynthesis?"*:

- **Base retriever** → Returns the full document (including Grand Canyon info)
- **Contextual compression** → Returns only *"Photosynthesis is the process by which green plants convert sunlight into energy."*

```python
compressor = LLMChainExtractor.from_llm(
    llm=ChatHuggingFace(llm=HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.1-70B-Instruct',
        task='text-generation',
    ))
)

contextual_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)
```

The script compares **compressed results** vs **raw retriever results** to show the difference.

---

## 🔄 Comparison at a Glance

| Retriever | Needs LLM? | Vector Store | Best For |
|-----------|:----------:|:------------:|----------|
| **Wikipedia** | ❌ | ❌ | Real-time factual queries |
| **Vector Store** | ❌ | ✅ Chroma | Simple similarity search |
| **MMR** | ❌ | ✅ FAISS | Diverse, non-redundant results |
| **Multi-Query** | ✅ | ✅ FAISS | Better recall via query expansion |
| **Contextual Compression** | ✅ | ✅ Chroma | Precise, noise-free extractions |

---

## 🛠️ Installation

```bash
# Core
pip install langchain langchain-community langchain-core langchain-classic

# Vector Stores
pip install langchain-chroma faiss-cpu

# Embeddings & LLMs
pip install langchain-google-genai langchain-huggingface

# Wikipedia
pip install wikipedia

# Environment
pip install python-dotenv
```

> 💡 Don't forget to set up your `.env` file with your API keys:
> ```env
> GOOGLE_API_KEY=your_google_api_key
> HUGGINGFACEHUB_API_TOKEN=your_hf_token
> ```

---

## ▶️ Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run any retriever
python wikipedia_retriever.py
python vectorStore_retriever.py
python mmr_retriever.py
python multiquery_retriever.py
python contexual_retriever.py
```

---

<p align="center">
  Built with 🦜🔗 <strong>LangChain</strong> | Embeddings by <strong>Google Gemini</strong> | LLM by <strong>Meta Llama</strong>
</p>
