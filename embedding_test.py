import numpy as np
from sentence_transformers import SentenceTransformer


texts =[
    "من برنامه نویسی رو دوست دارم",
    "من عاشق کدنویسی هستم",
    "امروز هوا بارانی است",
]



model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)
print("Vector dimension:", embeddings.shape[1])


def cosine_similarity(vector_a, vector_b):
    dot_product = vector_a @ vector_b
    norm_a = (vector_a @ vector_a) ** 0.5
    norm_b = (vector_b @ vector_b) ** 0.5

    return dot_product / (norm_a * norm_b)


similar_1 = cosine_similarity(embeddings[0], embeddings[1])
similar_2 = cosine_similarity(embeddings[0], embeddings[2])

print("Programming vs Coding:", similar_1)
print("Programming vs Weather:", similar_2)