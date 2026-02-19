import time
import numpy as np
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from transformers import pipeline


# Document Loader + Chunking
def load_and_split(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)

# Embedding Model
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Vector Store
def build_vector_store(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

# BM25 Retriever
class BM25Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.corpus = [doc.page_content.split() for doc in docs]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in top_indices]

# Hybrid Retriever
class HybridRetriever:
    def __init__(self, vectorstore, bm25):
        self.vectorstore = vectorstore
        self.bm25 = bm25

    def retrieve(self, query, top_k=5):
        vector_docs = self.vectorstore.similarity_search(query, k=top_k)
        bm25_docs = self.bm25.retrieve(query, top_k)

        combined = list({doc.page_content: doc for doc in vector_docs + bm25_docs}.values())
        return combined[:top_k]


# LLM Generator (Local HF Model)

generator = pipeline("text-generation", model="google/flan-t5-base")

def generate_answer(query, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
    Answer the question based only on the context below.
    Context:
    {context}

    Question: {query}
    """

    output = generator(prompt, max_length=300, do_sample=False)
    return output[0]["generated_text"]

