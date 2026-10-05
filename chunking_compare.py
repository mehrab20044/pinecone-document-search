import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def cosine_similarity(vector_a, vector_b):
    dot_product = vector_a @ vector_b
    norm_a = (vector_a @ vector_a) ** 0.5
    norm_b = (vector_b @ vector_b) ** 0.5

    return dot_product / (norm_a * norm_b)


with open("corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()


model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


sentences = [
    sentence.strip()
    for sentence in re.split(r"(?<=[.!?])\s+", text)
    if sentence.strip()
]

sentence_embeddings = model.encode(sentences)

semantic_chunks = []
current_chunk = sentences[0]

threshold = 0.45

for i in range(1, len(sentences)):
    similarity = cosine_similarity(
        sentence_embeddings[i - 1],
        sentence_embeddings[i],
    )

    if similarity >= threshold:
        current_chunk += " " + sentences[i]
    else:
        semantic_chunks.append(current_chunk)
        current_chunk = sentences[i]

semantic_chunks.append(current_chunk)


if __name__ == "__main__":
    print("Semantic chunks:", len(semantic_chunks))

    for index, chunk in enumerate(semantic_chunks):
        print(f"\nSemantic chunk {index}:")
        print(chunk)