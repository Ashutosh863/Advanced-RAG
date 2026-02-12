import time
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_answer(query, generated_answer, ground_truth):
    emb_gen = model.encode(generated_answer, convert_to_tensor=True)
    emb_gt = model.encode(ground_truth, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(emb_gen, emb_gt).item()

    return {
        "answer_similarity": round(similarity, 4)
    }

def measure_latency(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()

    return result, round(end - start, 3)
