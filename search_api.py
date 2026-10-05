from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from chunking_compare import semantic_chunks, cosine_similarity


app = FastAPI()

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

chunks_embeddings = model.encode(semantic_chunks)


@app.get("/search")
def search(q: str):
    query_embedding = model.encode(q)

    results = []

    for index, chunk_embedding in enumerate(chunks_embeddings):
        score = cosine_similarity(
            query_embedding,
            chunk_embedding,
        )

        results.append(
            {
                "score": float(score),
                "text": semantic_chunks[index],
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "query": q,
        "results": results[:3],
    }