# Advanced RAG + Evaluation Framework

## 🚀 Project Overview

This project implements an **Advanced Retrieval-Augmented Generation
(RAG)** system with an integrated evaluation framework.

It demonstrates:

-   Document ingestion (PDF)
-   Smart chunking
-   Vector embeddings (HuggingFace)
-   FAISS vector database
-   Hybrid retrieval (Vector + BM25)
-   LLM-based generation
-   Semantic evaluation using embedding similarity
-   Latency measurement

This project is designed for AI Engineers who want production-level
understanding of RAG systems.

------------------------------------------------------------------------

## 🧠 Architecture

User Query\
↓\
Hybrid Retriever (Vector + BM25)\
↓\
Context Augmentation\
↓\
LLM Generation\
↓\
Evaluation Module (Semantic Similarity)

------------------------------------------------------------------------

## 📁 Project Structure

advanced_rag/ │ ├── data/\
│ └── sample.pdf\
│ ├── rag_system.py\
├── evaluation.py\
├── app.py\
├── requirements.txt\
└── README.md

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment (Python 3.12 recommended)

Windows:

    py -3.12 -m venv venv
    venv\Scripts\activate

Mac/Linux:

    python3.12 -m venv venv
    source venv/bin/activate

------------------------------------------------------------------------

### 2️⃣ Install Dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

### 3️⃣ Add Your PDF

Place your PDF inside:

    data/sample.pdf

Or modify the path inside `app.py`.

------------------------------------------------------------------------

### 4️⃣ Run the Project

    python app.py

You will be prompted to:

1.  Enter a question related to the PDF\
2.  Enter the ground truth answer for evaluation

------------------------------------------------------------------------

## 📊 Evaluation Metrics

The system computes:

-   **Answer Similarity Score**
-   **Retrieval Latency**
-   **Generation Latency**

Similarity Score Interpretation:

-   0.90+ → Excellent
-   0.75--0.90 → Good
-   0.50--0.75 → Partial
-   \<0.50 → Likely incorrect

------------------------------------------------------------------------

## 🏗 What This Project Demonstrates

-   Practical RAG pipeline design
-   Embedding-based semantic evaluation
-   Hybrid search implementation
-   Modular architecture
-   Real-world debugging and dependency management

------------------------------------------------------------------------

## 🔥 Future Improvements

-   Cross-encoder reranking
-   Multi-query retrieval
-   HyDE retrieval
-   Streaming LLM responses
-   FastAPI backend integration
-   Evaluation dashboard
-   Docker deployment

------------------------------------------------------------------------

## 🎯 Learning Outcome

By completing this project, you gain:

-   Deep understanding of RAG internals
-   Retrieval tuning skills
-   Evaluation engineering knowledge
-   Production-ready AI backend experience

------------------------------------------------------------------------

## 👨‍💻 Author

Built as part of an AI Engineer learning roadmap.

------------------------------------------------------------------------
