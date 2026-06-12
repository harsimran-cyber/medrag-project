# MedRAG – Medical Symptom Checker (RAG-based)

MedRAG is a Retrieval-Augmented Generation (RAG) based medical symptom checker. It retrieves relevant medical information from a curated knowledge base and uses a local LLM to generate helpful, context-aware responses to user symptom queries.

> ⚠️ **Disclaimer:** This project is for educational purposes only and is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## 🚀 Features

- Retrieval-Augmented Generation (RAG) pipeline using LangChain
- Local LLM inference via Ollama (Mistral 7B)
- Semantic search powered by ChromaDB vector store
- Sentence embeddings using HuggingFace `all-MiniLM-L6-v2`
- Knowledge base built from NHS UK Conditions documentation
- Simple, interactive web UI built with Streamlit

---

## 🛠️ Tech Stack

| Component        | Technology                                  |
|-------------------|---------------------------------------------|
| Language          | Python 3.11                                  |
| RAG Framework     | LangChain (LCEL)                             |
| Vector Store      | ChromaDB                                     |
| Embeddings        | sentence-transformers/all-MiniLM-L6-v2       |
| LLM               | Mistral 7B (via Ollama)                      |
| UI                | Streamlit                                    |
| Document Loader   | PyPDF                                        |

---

## 📂 Project Structure

```
medrag-project/
├── data/              # Source PDF documents (NHS conditions)
├── chroma_db/         # Vector store (generated, not tracked)
├── ingest.py          # Script to load, chunk, and embed documents
├── app.py             # Streamlit application
├── requirements.txt   # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/harsimran-cyber/medrag-project.git
cd medrag-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model
Download Ollama from [ollama.com](https://ollama.com), then run:
```bash
ollama pull mistral
```

### 5. Ingest the knowledge base
```bash
python ingest.py
```

### 6. Run the application
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 🧠 How It Works

1. **Document Loading & Chunking** – NHS condition PDFs are loaded and split into 500-character chunks with 50-character overlap.
2. **Embedding & Storage** – Each chunk is embedded using `all-MiniLM-L6-v2` and stored in a ChromaDB vector database.
3. **Retrieval** – On a user query, the top-5 most relevant chunks are retrieved using cosine similarity.
4. **Generation** – The retrieved context is passed to Mistral 7B (via Ollama) to generate a final response using an LCEL chain.

---

## 🔮 Future Enhancements

- [ ] RAGAS-based evaluation framework
- [ ] Hybrid search (keyword + semantic)
- [ ] Expanded knowledge base beyond NHS sources
- [ ] Deployment beyond localhost

---

## 📄 License

This project is for academic purposes as part of a Deep Learning course at Thapar Institute of Engineering and Technology (TIET), Patiala.
