import numpy as np


# 1. DOT PRODUCT

def dot_product(vec1, vec2):
    result = 0.0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]
    return result



# 2. VECTOR MAGNITUDE

def magnitude(vec):
    sum_sq = 0.0
    for val in vec:
        sum_sq += val * val
    return np.sqrt(sum_sq)



# 3. COSINE SIMILARITY (MANUAL)

def cosine_similarity_manual(vec1, vec2):
    mag1 = magnitude(vec1)
    mag2 = magnitude(vec2)

    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product(vec1, vec2) / (mag1 * mag2)



# 4. COMPUTE SIMILARITY

def compute_similarity(query_vec, doc_vectors):
    similarities = []

    for vec in doc_vectors:
        sim = cosine_similarity_manual(query_vec, vec)
        similarities.append(sim)

    return similarities
