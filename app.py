from rag_system import *
from evaluation import *

# Load & Split
docs = load_and_split("data/sample.pdf")

# Create embeddings
embeddings = create_embeddings()

# Build vector DB
vectorstore = build_vector_store(docs, embeddings)

# BM25
bm25 = BM25Retriever(docs)

# Hybrid
retriever = HybridRetriever(vectorstore, bm25)

# Ask Question
query = input("Enter your question: ")

# Retrieval
retrieved_docs, retrieval_time = measure_latency(
    retriever.retrieve, query
)

print(f"\nRetrieval Time: {retrieval_time}s")

# Generation
answer, generation_time = measure_latency(
    generate_answer, query, retrieved_docs
)

print(f"\nGeneration Time: {generation_time}s")
print("\nGenerated Answer:\n")
print(answer)

# Evaluation (Manual ground truth for demo)
ground_truth = input("\nEnter ground truth answer (for evaluation): ")

metrics = evaluate_answer(query, answer, ground_truth)

print("\nEvaluation Metrics:")
print(metrics)
